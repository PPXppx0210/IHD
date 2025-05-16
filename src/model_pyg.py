from utils.log import logger

import numpy as np
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, GATv2Conv
from utils.Constants import PAD
from utils.utils import save_pickle
from src.model import AdditiveAttention, SpGATLayer, MultiHeadGraphAttention2
from typing import Dict, List, Tuple
from torch_geometric.utils import dropout_adj, to_dense_adj

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.to(seq.device)
    
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.to(seq.device)
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.to(seq.device)
    masked_seq = ans_tmp.scatter_(2,masked_seq.long(),float('-inf'))
    return masked_seq

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        # self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        print(self)

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
        init.xavier_normal_(self.linear3.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param mask: (*, max_q_words)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=0).bool().to(pad_mask.device)
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        '''
        :param Q:
        :param K:
        :param V:
        :param mask: (bsz, max_q_words)
        :return:
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V, mask=None):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V, mask)

        fusion_gate = torch.sigmoid(self.linear3(torch.cat([Q, V_att], dim=-1)))
        fusion = fusion_gate * V_att + (1 - fusion_gate) * Q

        if self.is_layer_norm:
            # X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            X = self.layer_morm(fusion)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output
    

class SpecialGraphNN(nn.Module):
    def __init__(self, ntoken, nhid, dropout=0.3):
        super(SpecialGraphNN, self).__init__()
        self.nhid = nhid
        self.ntoken = ntoken
        self.embedding = nn.Embedding(ntoken, nhid, padding_idx=0)
        init.xavier_normal_(self.embedding.weight)

        self.gpn = GPN(2, 2, nhid, nhid*2, nhid)
        self.drop = nn.Dropout(dropout)

    def forward(self, diffusion_graph):
        graph_edge_index = diffusion_graph.edge_index
        graph_weight = diffusion_graph.edge_weight
        graph_x_embeddings = self.gpn(self.embedding.weight, graph_edge_index, graph_weight)
        graph_x_embeddings = self.drop(graph_x_embeddings)
        return graph_x_embeddings 


class GCNNetwork(nn.Module):
    def __init__(
        self, n_feat, n_units, dropout, dropedge_adj=None):
        super(GCNNetwork, self).__init__()

        self.dropout = dropout
        self.dropout_adj = dropedge_adj
        self.layer_stack, self.batchnorm_stack = self._build_layer_stack(extend_units=[n_feat]+n_units,)
        # self.init_weights()

    def init_weights(self):
        for layer in self.layer_stack:
            init.xavier_normal_(layer.weight)
        for batchnorm in self.batchnorm_stack:
            init.xavier_normal_(batchnorm.weight)
    
    def _build_layer_stack(self, extend_units,):
        layer_stack = nn.ModuleList()
        batchnorm_stack = nn.ModuleList()
        for layer_i, (n_unit, f_out,) in enumerate(zip(extend_units[:-1], extend_units[1:],)):
            not_last_layer = layer_i != len(extend_units[:-1])-1
            layer_stack.append(
                GCNConv(in_channels=n_unit, out_channels=f_out, add_self_loops=True,),
            )
            if not not_last_layer:
                batchnorm_stack.append(nn.BatchNorm1d(f_out))
        return layer_stack, batchnorm_stack
    
    def forward(self, graph, emb):
        graph_edge_index = graph.edge_index
        graph_weight = graph.edge_weight
        if self.dropout_adj is not None:
            graph_edge_index, graph_weight = dropout_adj(graph_edge_index, graph_weight, p=self.dropout_adj, training=self.training)
        # raw_emb = emb.clone()
        fusion_emb = [emb.unsqueeze(2)]

        for layer_i, (gat_layer, batchnorm_layer) in enumerate(zip(self.layer_stack, self.batchnorm_stack)):
            emb = gat_layer(emb, graph_edge_index, graph_weight).float()
            if layer_i < len(self.layer_stack):
                emb = batchnorm_layer(emb)
                emb = F.elu(emb)
                emb = F.dropout(emb, self.dropout, training=self.training)
            fusion_emb.append(emb.unsqueeze(2))
        emb = self.layer_stack[-1](emb, graph_edge_index, graph_weight)
        fusion_emb.append(emb.unsqueeze(2))
        fusion_emb = torch.mean(torch.cat(fusion_emb, dim=2), dim=2)
        # fusion_emb = torch.mean(torch.cat([raw_emb.unsqueeze(2),emb.unsqueeze(2)], dim=2), dim=2)
        return fusion_emb

class GATNetwork(nn.Module):
    def __init__(
        self, n_feat, n_units, n_heads, attn_dropout, dropout,
    ):
        super(GATNetwork, self).__init__()

        self.dropout = dropout
        self.layer_stack, self.batchnorm_stack = self._build_layer_stack(extend_units=[n_feat]+n_units, n_heads=n_heads, attn_dropout=attn_dropout)
        # self.init_weights()

    def init_weights(self):
        for layer in self.layer_stack:
            init.xavier_normal_(layer.weight)
        for batchnorm in self.batchnorm_stack:
            init.xavier_normal_(batchnorm.weight)
    
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout):
        layer_stack = nn.ModuleList()
        batchnorm_stack = nn.ModuleList()
        for layer_i, (n_unit, n_head, f_out, fin_head) in enumerate(zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1])):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            not_last_layer = layer_i != len(extend_units[:-1])-1
            layer_stack.append(
                GATv2Conv(heads=n_head, in_channels=f_in, out_channels=f_out, concat=True, dropout=attn_dropout, add_self_loops=True, edge_dim=1),
                # SpGATLayer(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
            if not_last_layer:
                batchnorm_stack.append(nn.BatchNorm1d(f_out*n_head))
        return layer_stack, batchnorm_stack
    
    def forward(self, graph, emb):
        graph_edge_index = graph.edge_index
        graph_weight = graph.edge_weight
        fusion_emb = [emb.unsqueeze(2)]

        for layer_i, (gat_layer, batchnorm_layer) in enumerate(zip(self.layer_stack, self.batchnorm_stack)):
            emb = gat_layer(emb, graph_edge_index, graph_weight)
            if layer_i < len(self.layer_stack):
                emb = batchnorm_layer(emb)
                emb = F.elu(emb)
                emb = F.dropout(emb, self.dropout, training=self.training)
            fusion_emb.append(emb.unsqueeze(2))
        emb = self.layer_stack[-1](emb, graph_edge_index, graph_weight)
        fusion_emb.append(emb.unsqueeze(2))
        fusion_emb = torch.mean(torch.cat(fusion_emb, dim=2), dim=2)
        return fusion_emb

class GATNetwork2(nn.Module):
    def __init__(self, n_units, n_heads, attn_dropout, dropout,):
        super(GATNetwork2, self).__init__()
        self.dropout = dropout
        self.layer_stack, self.batchnorm_stack = self._build_layer_stack(n_units=n_units, n_heads=n_heads, attn_dropout=attn_dropout)
    
    def _build_layer_stack(self, n_units, n_heads, attn_dropout)->Tuple[List[MultiHeadGraphAttention2],List[nn.BatchNorm1d]]:
        layer_stack = nn.ModuleList()
        batchnorm_stack = nn.ModuleList()
        for layer_i, (n_feat, n_head,) in enumerate(zip(n_units, n_heads)):
            layer_stack.append(MultiHeadGraphAttention2(n_head=n_head, feat=n_feat, attn_dropout=attn_dropout, bias=True),)
            if layer_i < len(n_units)-1:
                batchnorm_stack.append(nn.BatchNorm1d(n_feat*n_head))
        return layer_stack, batchnorm_stack
    
    def forward(self, adj:torch.Tensor, seq_embs:torch.Tensor, time_decay_emb:torch.Tensor, pad_mask:torch.Tensor=None):
        # adj:(bs,n,n), seq_embs:(bs,n,n_head,feat), time_decay:(bs,n)
        bs, n = seq_embs.size()[:2]
        fusion_emb = [seq_embs.contiguous().view(bs,n,-1).unsqueeze(-1)]

        for layer_i, gat_layer in enumerate(self.layer_stack,):
            emb = gat_layer(query=seq_embs, adj=adj, decay_emb=time_decay_emb, pad_mask=pad_mask) # (bs,n,n_head*feat)
            if torch.isnan(emb).any():
                print(emb)
                print(f"NaN detected in layer {layer_i}!")
                print("Input stats:", seq_embs.min(), seq_embs.max())
                print("Adj stats:", adj.min(), adj.max())
                raise ValueError("NaN in GAT layer output")
            if layer_i < len(self.layer_stack)-1:
                # emb = F.elu(self.batchnorm_stack[layer_i](emb.transpose(1,2)).transpose(1,2))
                emb = F.dropout(F.elu(emb), self.dropout, training=self.training)
            fusion_emb.append(emb.unsqueeze(-1)) # (bs,n,n_head*feat,1)
        fusion_emb = torch.mean(torch.cat(fusion_emb, dim=-1), dim=-1) # (bs,n,n_head*feat,L)->(bs,n,n_head*feat)
        return fusion_emb

class TimeAttention_New(nn.Module):
    def __init__(self, ninterval, nfeat, dropout=0.1):
        super(TimeAttention_New, self).__init__()
        self.time_embedding = nn.Embedding(ninterval, nfeat)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cas_intervals, cas_embs, mask=None, episilon=1e-6):
        temperature = cas_embs.size(-1) ** 0.5 + episilon # d**0.5+eps
        cas_interval_embs = self.time_embedding(cas_intervals)

        affine = torch.einsum("bqd,bkd->bqk", cas_embs, cas_interval_embs)
        score = affine / temperature

        pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, mask.size(1))
        mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool()
        if pad_mask.is_cuda:
            mask = mask.to(pad_mask.device)
        mask_ = mask + pad_mask
        score = score.masked_fill(mask_, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        return alpha.bmm(cas_embs)

class BasicGATNetwork(nn.Module):
    def __init__(self, n_feat, n_units, n_heads, num_interval, shape_ret,
        attn_dropout, dropout, instance_normalization=False, use_gat=False
    ):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(BasicGATNetwork, self).__init__()

        self.dropout = dropout
        self.inst_norm = instance_normalization
        
        self.user_size = shape_ret[1]
        self.user_emb = nn.Embedding(self.user_size, n_feat, padding_idx=PAD)
        if use_gat:
            self.gat_network = GATNetwork(n_feat, n_units, n_heads, attn_dropout, dropout)
        else:
            self.gat_network = GCNNetwork(n_feat, n_units, dropout)
        self.time_attention = TimeAttention_New(ninterval=num_interval, nfeat=shape_ret[0])
        self.fc_network = nn.Linear(shape_ret[0], shape_ret[1])
        self.init_weights()
    
    def init_weights(self):
        init.xavier_normal_(self.fc_network.weight)
    
    def forward(self, cas_uids, cas_intervals, graph):
        cas_uids = cas_uids[:,:-1]
        cas_intervals = cas_intervals[:,:-1]

        user_emb = self.user_emb(torch.tensor([i for i in range(self.user_size)]).to(cas_uids.device))
        graph_emb = self.gat_network(graph, user_emb)
        seq_embs = F.embedding(cas_uids, graph_emb) # (bs, max_len, D)

        mask = (cas_uids == PAD)
        seq_embs = self.time_attention(cas_intervals, seq_embs, mask)
        seq_embs = F.dropout(seq_embs, self.dropout)

        output = self.fc_network(seq_embs)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output

class DiffusionGATNetwork(nn.Module):
    def __init__(self, n_interval, shape_ret, dropout,):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(DiffusionGATNetwork, self).__init__()

        ntoken = shape_ret[1]
        ninp = shape_ret[0]
        self.ninp = ninp
        self.user_size = ntoken

        self.pos_dim = 8
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)

        self.dropout = nn.Dropout(dropout)
        self.drop_timestamp = nn.Dropout(dropout)

        self.gnn_diffusion = SpecialGraphNN(ntoken, ninp)
        self.time_attention = TimeAttention_New(n_interval, self.ninp + self.pos_dim)
        self.decoder_attention = TransformerBlock(input_size=ninp + self.pos_dim, n_heads=8)
        self.linear = nn.Linear(ninp + self.pos_dim, ntoken)
        self.init_weights()
    
    def init_weights(self):
        # init.xavier_normal_(self.user_emb.weight)
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)
    
    def forward(self, cas_uids, cas_tss, diffusion_graph):
        cas_uids = cas_uids[:,:-1]
        cas_tss = cas_tss[:,:-1]
        mask = (cas_uids == PAD)

        batch_t = torch.arange(cas_uids.size(1)).expand(cas_uids.size()).to(cas_uids.device)
        order_embed = self.dropout(self.pos_embedding(batch_t))

        latest_timestamp = sorted(diffusion_graph.keys())[-1]
        graph_dynamic_embeddings = self.gnn_diffusion(diffusion_graph[latest_timestamp])
        dyemb = F.embedding(cas_uids, graph_dynamic_embeddings.to(cas_uids.device))
        dyemb = self.dropout(dyemb)

        final_embed = torch.cat([dyemb, order_embed], dim=-1) # dynamic_node_emb

        # final_embed = self.time_attention(dyemb_timestamp.to(input.device), final_embed, mask)
        final_embed = self.time_attention(cas_tss.to(cas_uids.device), final_embed, mask)

        att_out = self.decoder_attention(final_embed, final_embed, final_embed, mask=mask)

        att_out = self.dropout(att_out)
        output = self.linear(att_out)  # (bsz, user_len, |U|)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask

        return output.view(-1, output.size(-1))

class LHD(nn.Module):
    def __init__(self, user_size, n_feat, n_adj, num_interval, n_comp,
        n_units, n_heads, attn_dropout, dropout, instance_normalization=False, 
        # Ablation
        use_gat=True, use_time_decay=False, 
        use_add_attn=False, use_topic_selection=True, random_feat_dim=None,
    ):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(LHD, self).__init__()

        self.n_heads = n_heads
        # self.n_heads = n_heads
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.user_size = user_size
        self.user_emb = nn.Embedding(self.user_size, n_feat, padding_idx=PAD)
        pos_emb_dim = 8
        self.pos_emb  = nn.Embedding(1000, pos_emb_dim) # max_cascade_len (500) * pos_emb_dim

        # Model
        last_dim = n_units[-1] if not use_gat else n_units[-1]*n_heads[-1]
        last_dim_with_pos = last_dim + pos_emb_dim
        self.time_attention = TimeAttention_New(ninterval=num_interval, nfeat=last_dim_with_pos)
        self.decoder_attention = TransformerBlock(input_size=last_dim_with_pos, n_heads=8)
        self.fc_network = nn.Linear(last_dim_with_pos, user_size)
        self.graph_mlp = nn.Sequential(
            nn.Linear(last_dim * 2, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, 1)
        )
        self.init_weights()
        
        # Ablation
        self.use_gat = use_gat
        if self.use_gat:
            self.heter_gat_network = nn.ModuleList([GATNetwork2(n_units, n_heads, attn_dropout, dropout) for _ in range(n_adj)])
        else: # otherwise use gcn
            self.heter_gat_network = nn.ModuleList([GCNNetwork(n_feat, n_units, dropout) for _ in range(n_adj)])

        self.use_time_decay = use_time_decay
        if self.use_time_decay:
            self.time_decay_emb = nn.Embedding(num_interval, n_adj)
        
        self.use_add_attn = use_add_attn
        if self.use_add_attn:
            self.additive_attention = AdditiveAttention(d=n_feat, d1=last_dim, d2=last_dim)
            # self.fc_attn = nn.Linear(n_units[-1]*n_adj, n_units[-1])
        
        self.use_topic_selection = use_topic_selection

        self.use_random_feat = random_feat_dim is None
        if not self.use_random_feat:
            self.feat_fc_layer = nn.Linear(random_feat_dim, n_feat)

    def init_weights(self):
        # if not self.use_topic_pref:
        #     init.xavier_normal_(self.fc_topic_net.weight)
        init.xavier_normal_(self.fc_network.weight)
    
    def select_topics(self, aware_seq_embs:torch.Tensor, cas_classids:torch.Tensor,):
        '''
        args: 
            aware_seq_embs: (bs,ml,K*D),
        return: (bs,ml,n_comp*D),
        '''
        bs, ml, _, feat_dim = aware_seq_embs.size()
        n_comp = cas_classids.size(1)
        selected_aware_seq_embs = torch.zeros(bs, ml, n_comp, feat_dim) # (bs, max_len, n_comp, D')
        if aware_seq_embs.is_cuda:
            selected_aware_seq_embs = selected_aware_seq_embs.to(aware_seq_embs.device)
        for batch_i in range(bs):
            selected_aware_seq_embs[batch_i] = aware_seq_embs[batch_i, :, cas_classids[batch_i], :]
        selected_aware_seq_embs = F.dropout(selected_aware_seq_embs, self.dropout)
        return selected_aware_seq_embs
    
    def forward(self, cas_uids:torch.Tensor, cas_intervals:torch.Tensor, cas_classids:torch.Tensor, hedge_graphs, cas_tss=None, feats:torch.Tensor=None, multi_deepwalk_feat:torch.Tensor=None):
        # remove last user in each sequence
        cas_uids = cas_uids[:,:-1]
        cas_intervals = cas_intervals[:,:-1]
        if cas_tss is not None:
            cas_tss = cas_tss[:,:-1]
        # assert len(hedge_graphs) == len(self.heter_gat_network)
        # assert multi_deepwalk_feat.size() == [self.user_size, self.user_emb.embedding_dim//self.n_head, len(hedge_graphs)]

        # prepare input embeddings
        if self.use_random_feat:
        # if feats is None:
            user_emb2 = self.user_emb(torch.tensor([i for i in range(self.user_size)]).to(cas_uids.device))
        else:
            user_emb2 = self.feat_fc_layer(feats)

        # use graph neural networks
        bs, ml = cas_uids.size()[:2]
        if not self.use_gat: # use gcn
            heter_user_embs = []
            for heter_i, gat_network in enumerate(self.heter_gat_network):
                if multi_deepwalk_feat is None:
                    graph_emb = gat_network(hedge_graphs[heter_i], user_emb2)
                else:
                    graph_emb = gat_network(hedge_graphs[heter_i], multi_deepwalk_feat[:,:,heter_i].unsqueeze(1).expand(-1,self.n_heads[0],-1).reshape(self.user_size,-1))
                heter_user_embs.append(graph_emb.unsqueeze(1))
            topic_aware_embs = torch.cat(heter_user_embs, dim=1)
            aware_seq_embs = F.embedding(cas_uids, topic_aware_embs.reshape(self.user_size,-1)).reshape(bs,ml,-1,topic_aware_embs.size(-1)) # (bs, max_len, |Rs|+1, D')
        else: # NOTE: Use GATNetwork2 with time_decay
            if self.use_time_decay:
                time_decay_emb = self.time_decay_emb(cas_intervals) # (bs,ml,n_adj,d)
            seq_embs = F.embedding(cas_uids, user_emb2).reshape(bs,ml,self.n_heads[0],-1)

            # if multi_deepwalk_feat is not None:
            #     full_seq_embs = F.embedding(cas_uids, multi_deepwalk_feat.reshape(self.user_size,-1)).unsqueeze(2).expand(-1,-1,self.n_head,-1).reshape(bs,ml,self.n_head,len(hedge_graphs),-1)

            # Calculate Pad Mask & Pos Mask(*)
            pad_mask = (cas_uids == PAD).unsqueeze(1).expand(-1,ml,-1).clone() # (bs,ml,ml)
            # NOTE: Set Diagnol Elements to Zero, In Case PAD-MASK and ADJ-MASK masked all values, and thus get nan!!!
            ind = np.diag_indices(ml)
            pad_mask[:,ind[0],ind[1]] = torch.zeros(ml, dtype=pad_mask.dtype).to(pad_mask.device)
            pad_mask = pad_mask.unsqueeze(1) # (bs,1,n,n)
            # tri_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool()
            # if pad_mask.is_cuda:
            #     tri_mask = tri_mask.to(pad_mask.device)
            # pad_mask = pad_mask + tri_mask

            heter_user_embs = []
            for heter_i, gat_network in enumerate(self.heter_gat_network):
                graph_adj = to_dense_adj(edge_index=hedge_graphs[heter_i].edge_index, edge_attr=hedge_graphs[heter_i].edge_weight).squeeze() # (1,N,N)->(N,N)
                graph_adj[graph_adj!=0] = 1.
                zero_row_indices = torch.where(graph_adj.sum(dim=1) == 0)[0]
                batch_adjs = graph_adj[cas_uids.unsqueeze(1), cas_uids.unsqueeze(2)] # (N,N) -> (bs,ml,ml)
                # if multi_deepwalk_feat is not None:
                #     seq_embs = full_seq_embs[:,:,:,heter_i,:]
                if self.use_time_decay:
                    graph_emb = gat_network(adj=batch_adjs, seq_embs=seq_embs, time_decay_emb=time_decay_emb[:,:,heter_i], pad_mask=pad_mask) # (bs,ml,n_head*feat)
                else:
                    graph_emb = gat_network(adj=batch_adjs, seq_embs=seq_embs, time_decay_emb=None, pad_mask=pad_mask) # (bs,ml,n_head*feat)
                heter_user_embs.append(graph_emb.unsqueeze(-2))
            aware_seq_embs = torch.cat(heter_user_embs, dim=-2) # (bs,ml,K,D')
        
        user_seq_embs = F.embedding(cas_uids, user_emb2)
        # use preference fusion
        if self.use_add_attn:
            # user_seq_embs = F.embedding(cas_uids, user_emb2)
            user_seq_embs = user_seq_embs.view(-1,user_seq_embs.size(-1)) # (bs*max_len, D)
            aware_seq_embs = aware_seq_embs.view(bs*ml,-1,aware_seq_embs.size(-1)) # (bs*max_len, n_comp, D')
            assert user_seq_embs.size(0) == aware_seq_embs.size(0)
            fusion_seq_embs = self.additive_attention(user_seq_embs, aware_seq_embs) # (bs*max_len, 1, D')
            # NOTE: use fusion_seq_embs instead of mean pooling
            aware_seq_embs = torch.cat((aware_seq_embs, fusion_seq_embs),dim=1)
            aware_seq_embs = aware_seq_embs.reshape(bs, ml, -1, aware_seq_embs.size(-1)) # (bs, max_len, n_comp+1, D')
        # aware_seq_embs = torch.mean(aware_seq_embs, dim=2)
        K = aware_seq_embs.size(2)
        user_seq_exp = user_seq_embs.unsqueeze(2).expand(-1, -1, K, -1)  # (bs, ml, K, D)
        fusion_input = torch.cat([aware_seq_embs, user_seq_exp], dim=-1)  # (bs, ml, K, 2D)
        # reshape for MLP
        bs, ml, k, d2 = fusion_input.size()
        fusion_input = fusion_input.view(-1, d2)  # (bs * ml * K, 2D)
        # MLP to get attention weights
        weights = self.graph_mlp(fusion_input).view(bs * ml, k)  # (bs * ml, K)
        weights = F.softmax(weights, dim=-1)
        # apply weights
        aware_seq_embs = aware_seq_embs.view(bs * ml, k, -1)  # (bs * ml, K, D)
        weighted_sum = torch.bmm(weights.unsqueeze(1), aware_seq_embs).squeeze(1)  # (bs * ml, D)
        aware_seq_embs = weighted_sum.view(bs, ml, -1)  # (bs, ml, D)
        
        seq_embs = F.dropout(aware_seq_embs, self.dropout)

        batch_t = torch.arange(cas_uids.size(1)).expand(cas_uids.size()).to(cas_uids.device)
        pos_embs = F.dropout(self.pos_emb(batch_t), self.dropout)
        seq_embs = torch.cat([seq_embs, pos_embs], dim=-1)

        mask = (cas_uids == PAD)
        # seq_embs = self.time_attention(cas_intervals, torch.cat([seq_embs, pos_embs], dim=-1), mask)
        # seq_embs = F.dropout(seq_embs, self.dropout)

        seq_embs = self.decoder_attention(seq_embs, seq_embs, seq_embs, mask)
        output = self.fc_network(seq_embs) # (bs, max_len, |V|)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output
