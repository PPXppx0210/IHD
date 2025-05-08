from utils.log import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from dgl.nn.pytorch import GATConv, GATv2Conv
from typing import List, Union

class FullyConnectedLayer(nn.Module):
    def __init__(self, nb_input, nb_hidden, nb_output) -> None:
        super().__init__()

        self.fc1 = nn.Linear(nb_input, nb_hidden, bias=True)
        self.fc2 = nn.Linear(nb_hidden, nb_output, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()

        self.n_head = n_head
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = h.size(0) # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w) #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src) # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_dst) # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1) # n_head x n x n

        attn = self.leaky_relu(attn)
        attn.data.masked_fill_(~adj, float("-inf"))
        attn = self.softmax(attn) # n_head x n x n
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime).transpose(0,1) # n x n_head x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output

def check_all_inf(attn_weights, label):
    bs_nh, n, _ = attn_weights.shape
    all_inf_rows = (attn_weights == float('-inf')).all(dim=-1)
    # bs_nh, n = all_inf_rows.shape
    # for idx in range(bs_nh):
    #     for i in range(n):
    #         if all_inf_rows[idx, i]:
    #             print(f"[BatchHead={idx}, Node={i}] 全为-inf")
    num_all_inf_rows = all_inf_rows.sum().item()
    if num_all_inf_rows > 0:
        print(f"[{label}] {num_all_inf_rows}/{bs_nh * n} rows are all -inf")
        return True
    return False
        
class MultiHeadGraphAttention2(nn.Module):
    def __init__(self, n_head, feat, attn_dropout, bias=True,):
        super(MultiHeadGraphAttention2, self).__init__()
        self.dropout = attn_dropout
        self.feat = feat
        self.n_head = n_head
        self.scaling:int = feat ** -0.5
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.in_proj_weight = nn.Parameter(torch.empty(n_head, feat, 2*feat))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(n_head, 2*feat))
        else:
            self.register_parameter('in_proj_bias', None)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for i in range(self.n_head):
            init.xavier_uniform_(self.in_proj_weight[i,:,(self.feat*0):(self.feat*1)])
            init.xavier_uniform_(self.in_proj_weight[i,:,(self.feat*1):(self.feat*2)])
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.)
    
    def _in_proj_qkv(self, input:torch.Tensor):
        bs, n, n_head = input.size()[:3]
        input = input.contiguous().view(bs*n,n_head,-1).transpose(0,1) # (n_h,bs*n,feat)
        output = torch.matmul(input, self.in_proj_weight) # (n_h,bs*n,feat)*(n_h,feat,2*feat)->(n_h,bs*n,2*feat)
        if self.in_proj_bias is not None:
            output += self.in_proj_bias.unsqueeze(1)
        output = output.transpose(0,1).contiguous().view(bs,n,n_head,-1).transpose(1,2).contiguous().view(bs*n_head,n,-1)
        return output.chunk(2, dim=-1) # (bs*n_head,n,feat)*2
    
    def forward(self, query:torch.Tensor, adj:torch.Tensor=None, decay_emb:torch.Tensor=None, pad_mask:torch.Tensor=None):
        # query: (bs,n,n_head,feat), adj: (bs,n,n), decay_emb: (bs,n), pad_mask: (bs,1,n,n,)
        bs, n, n_head = query.size()[:3]
        v = query
        q, k = self._in_proj_qkv(query)
        q = q * self.scaling
        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) # (bs*n_head, n, n)
        attn_output_weights = self.leaky_relu(attn_output_weights)
        
        if adj is not None:
            attn_output_weights = attn_output_weights.view(bs,n_head,n,n).masked_fill(~adj.bool().unsqueeze(1), float('-inf')).view(bs*n_head,n,n)
            # attn_output_weights = attn_output_weights.view(bs,n_head,n,n).masked_fill(~adj.T.bool().unsqueeze(1), float('-inf')).view(bs*n_head,n,n)
        
        # if adj is not None:
        #     attn_output_weights = attn_output_weights.view(bs, n_head, n, n)
        #     attn_output_weights = attn_output_weights.masked_fill(~adj.bool().unsqueeze(1), float('-inf'))
        #     if check_all_inf(attn_output_weights.view(bs*n_head, n, n), "After adj mask"):
        #         print("→ Check your adjacency matrix — is it completely zero for some nodes?")
        #     attn_output_weights = attn_output_weights.view(bs * n_head, n, n)

        if decay_emb is not None:
            decay_mask = torch.bmm(decay_emb.unsqueeze(2), decay_emb.unsqueeze(1)) # (bs,n,n)
            decay_mask = decay_mask.unsqueeze(1).expand(-1,n_head,-1,-1).reshape(bs*n_head,n,n)
            attn_output_weights += decay_mask
        
        if pad_mask is not None:
            attn_output_weights = attn_output_weights.view(bs,n_head,n,n).masked_fill(pad_mask, float('-inf')).view(bs*n_head,n,n)

        # if pad_mask is not None:
        #     attn_output_weights = attn_output_weights.view(bs, n_head, n, n)
        #     attn_output_weights = attn_output_weights.masked_fill(pad_mask, float('-inf'))
        #     if check_all_inf(attn_output_weights.view(bs*n_head, n, n), "After pad_mask"):
        #         print("→ Padding mask might be too aggressive?")
        #     attn_output_weights = attn_output_weights.view(bs * n_head, n, n)
        
        # attn_output_weights = attn_output_weights.view(bs, n_head, n, n)
        # all_inf = (attn_output_weights == float('-inf')).all(dim=-1, keepdim=True)  # shape: (bs, n_head, n, 1)
        # attn_output_weights = attn_output_weights.masked_fill(all_inf.expand(-1, -1, -1, n), 0.0)
        # eye = torch.eye(n, device=attn_output_weights.device).unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)
        # attn_output_weights = attn_output_weights.masked_fill(all_inf.expand(-1, -1, -1, n) & eye.bool(), 1.0)
        # attn_output_weights = attn_output_weights.view(bs*n_head, n, n)

        attn_output_weights = F.softmax(attn_output_weights.float(), dim=-1, dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v.transpose(1,2).contiguous().view(bs*n_head,n,-1)).view(bs,n_head,n,-1).transpose(1,2) # (bs*n_head,n,feat) -> (bs,n,n_head,feat)
        return attn_output.contiguous().view(bs,n,-1) # (bs,n,n_head*feat)

class SpGATLayer(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=False):
        super(SpGATLayer, self).__init__()

        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out

        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.attn_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.attn_trg = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=attn_dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_trg)

    def explicit_broadcast(self, this, other):
        """from https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb"""
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        
        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    
    def special_spmm(self, other, trg_indices, size):
        ret = torch.zeros(size=size, dtype=other.dtype, device=other.device)
        trg_indices_broadcast = self.explicit_broadcast(trg_indices, other)
        ret.scatter_add_(dim=0, index=trg_indices_broadcast, src=other)
        return ret
    
    def calc_neigh_attn(self, e, trg_indices, nb_nodes):
        e = e - e.max()
        exp_e = e.exp()
        neigh_attn = exp_e / (self.special_spmm(other=exp_e, trg_indices=trg_indices, size=(nb_nodes, self.n_head)).index_select(0, trg_indices)+1e-16)
        return neigh_attn.unsqueeze(-1) # (E,k)->(E,k,1)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # NOTE: h: (n, fin), adj: (n, n)
        n = h.shape[0]
        h_prime  = torch.matmul(h.unsqueeze(0), self.w) # (k,n,f_out)
        h_prime  = self.dropout(h_prime)
        attn_src = torch.matmul(h_prime, self.attn_src).sum(dim=-1).permute(1,0) # (n,k)
        attn_trg = torch.matmul(h_prime, self.attn_trg).sum(dim=-1).permute(1,0) # (n,k)

        (src_indices, trg_indices) = adj._indices()
        attn_src_lifted, attn_trg_lifted, h_prime_lifted = attn_src[src_indices], attn_trg[trg_indices], h_prime.permute(1,0,2)[src_indices]
        e = self.leaky_relu(attn_src_lifted+attn_trg_lifted) # (E,k)

        neigh_attn = self.calc_neigh_attn(e=e, trg_indices=trg_indices, nb_nodes=n) # (E,k,1)
        neigh_attn = self.dropout(neigh_attn)

        h_weighted = h_prime_lifted*neigh_attn # (E,k,f_out)
        out = self.special_spmm(h_weighted, trg_indices=trg_indices, size=(n, self.n_head, self.f_out)) # (n,k,f_out)

        if self.bias is not None:
            return out + self.bias
        else:
            return out

class DenseSparseGAT(nn.Module):
    def __init__(
        self, static_nfeat, dynamic_nfeats, n_units, n_heads, shape_ret,
        attn_dropout, dropout, instance_normalization=False,
    ) -> None:
        """
        Arguments:
            n_units: contains hidden unit dimension of each layer
            n_heads: contains attention head number of each layer
            shape_ret: (n_user,nb_classes=2), contains shape of required return tensor
        """
        super().__init__()

        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            for i, dynamic_nfeat in enumerate(dynamic_nfeats):
                norm = nn.InstanceNorm1d(dynamic_nfeat, momentum=0.0, affine=True)
                setattr(self, f"norm-{i}", norm)

        n_feat = static_nfeat + sum(dynamic_nfeats)
        self.layer_stack = self._build_layer_stack(extend_units=[n_feat]+n_units, n_heads=n_heads, attn_dropout=attn_dropout)
        # self.fc_layer = nn.Linear(in_features=n_units[-1], out_features=shape_ret[1])
    
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                SpGATLayer(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
        return layer_stack

    def forward(self, adj:torch.Tensor, static_emb:torch.Tensor, dynamic_embs:List[torch.Tensor]):
        # adj: n*n, emb: n*n_feat
        norm_embs = []
        for i, dynamic_emb in enumerate(dynamic_embs):
            norm_emb = dynamic_emb
            if self.inst_norm:
                norm = getattr(self, f"norm-{i}")
                norm_emb = norm(norm_emb.transpose(0,1)).transpose(0,1)
            norm_embs.append(norm_emb)
        emb = torch.cat(norm_embs, dim=1)
        emb = torch.cat((emb,static_emb),dim=1)

        n = adj.shape[0]
        for i, gat_layer in enumerate(self.layer_stack):
            emb = gat_layer(emb, adj) # n_head*n*f_out
            if i+1 == len(self.layer_stack):
                emb = emb.mean(dim=1) # n*f_out
            else:
                emb = F.elu(emb.reshape(n, -1)) # n*(n_head*f_out)
                emb = F.dropout(emb, self.dropout, training=self.training)
        # emb = self.fc_layer(emb) # bs*n*shape_ret[1]
        return F.log_softmax(emb, dim=-1)

class AdditiveAttention(nn.Module):
    def __init__(self, d, d1, d2) -> None:
        super().__init__()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.w1 = nn.Linear(in_features=d, out_features=d2, bias=False)
        self.w2 = nn.Linear(in_features=d1, out_features=d2, bias=False)
        self.m  = nn.Linear(in_features=d2, out_features=1, bias=False)
    
    def forward(self, feature: torch.Tensor, type_aware_emb: torch.Tensor):
        """
        Args:
            feature: initial feature of all main nodes, shape is (N, D)
            type_aware_emb: ~, shape is (N, |Rs|, D')
        Return:
            type_fusion_emb: ~, shape is (N, 1, D')
        """
        _, nb_node_kind, _ = type_aware_emb.shape
        feature = feature.unsqueeze(-2).repeat(1, nb_node_kind, 1) # (N, |Rs|, D)

        q = self.tanh(self.w1(feature)+self.w2(type_aware_emb)) # (N, |Rs|, D'')
        q = self.m(q) # (N, |Rs|, 1)
        beta = self.softmax(q.squeeze(-1)) # (N, |Rs|)
        type_fusion_emb = torch.bmm(
            beta.unsqueeze(-2), type_aware_emb #(N,1,|Rs|) * (N,|Rs|,D') -> (N,1,D')
        )
        return type_fusion_emb

class HeterEdgeSparseGAT(nn.Module):
    """
    思路: 多邻接矩阵+稀疏注意力头
    特点: 异质边, 节点类型只有用户这一类
    """
    def __init__(
        self, static_nfeat, dynamic_nfeats, n_adj, n_units, n_heads, shape_ret,
        attn_dropout, dropout, instance_normalization=False, sparse=True,
    ) -> None:
        """
        Arguments:
            n_adj:   implied how much heterogeneous adj matrices there are
            n_units: contains hidden unit dimension of each layer
            n_heads: contains attention head number of each layer
            shape_ret: (n_user,nb_classes=2), contains shape of required return tensor
        """
        super().__init__()

        self.shape_ret = shape_ret
        self.dropout = dropout
        self.inst_norm = instance_normalization
        d1 = n_units[-1]

        if self.inst_norm:
            for i, dynamic_nfeat in enumerate(dynamic_nfeats):
                norm = nn.InstanceNorm1d(dynamic_nfeat, momentum=0.0, affine=True)
                setattr(self, f"norm-{i}", norm)

        n_feat = static_nfeat + sum(dynamic_nfeats)
        self.layer_stack = nn.ModuleList([
                self._build_layer_stack(extend_units=[n_feat]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse)
            for _ in range(n_adj)])
        self.additive_attention = AdditiveAttention(d=n_feat, d1=d1, d2=n_units[-1])
        self.fc_layer = nn.Linear(in_features=d1*(n_adj+1), out_features=shape_ret[1])

    def _build_layer_stack(self, extend_units, n_heads, attn_dropout, sparse):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                SpGATLayer(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout) if sparse else
                    MultiHeadGraphAttention(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
        return layer_stack
    
    def forward(self, hadjs:List[torch.Tensor], static_emb:torch.Tensor, dynamic_embs:List[torch.Tensor]): # hadjs:(|Rs|,n,n), embs:(|Rs|,n,f_in)
        norm_embs = []
        for i, dynamic_emb in enumerate(dynamic_embs):
            norm_emb = dynamic_emb
            if self.inst_norm:
                norm = getattr(self, f"norm-{i}")
                norm_emb = norm(norm_emb.transpose(0,1)).transpose(0,1)
            norm_embs.append(norm_emb)
        emb = torch.cat(norm_embs, dim=1)
        emb = torch.cat((emb,static_emb),dim=1)

        heter_embs = []
        for heter_idx, layer_stack in enumerate(self.layer_stack):
            x, hadj = emb, hadjs[heter_idx] # x: (n,f_in_unified)
            n = x.shape[0]
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(x, hadj) # (n,n_head,f_out)
                if i+1 == len(layer_stack):
                    x = x.mean(dim=1) # (n,n_head,f_out) -> (n,f_out)
                else:
                    x = F.elu(x.reshape(n, -1)) # (n,n_head,f_out) -> (n,n_head*f_out)
                    x = F.dropout(x, self.dropout, training=self.training)
            heter_embs.append(x[:self.shape_ret[0]].unsqueeze(-2)) # (Nu, 1, f_out)
        type_aware_emb = torch.cat(heter_embs, dim=-2) # (Nu, |Rs|, D')
        type_fusion_emb = self.additive_attention(emb, type_aware_emb) # (Nu, 1, D')
        ret = self.fc_layer(
            torch.cat((type_aware_emb, type_fusion_emb),dim=1).reshape(self.shape_ret[0],-1) # (Nu, |Rs|+1, D') -> (Nu, (|Rs|+1)*D')
        ) #  (Nu, nb_classes)
        return F.log_softmax(ret, dim=-1)
