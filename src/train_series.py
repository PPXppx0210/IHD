# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.log import logger
from utils.utils import *
from utils.graph import build_heteredge_mats
from utils.graph_aminer import *
from utils.metric import compute_metrics
from utils.Constants import PAD, EOS
from utils.Optim import ScheduledOptim
from utils.Patience import EarlyStopping
from src.graph_learner import *
from src.data_loader import DataConstruct
from src.model_pyg import *
from src.sota.TAN.model import TAN
from src.sota.TAN.Option import Option
from src.sota.DHGPNTM.DyHGCN import DyHGCN_H
from src.sota.DHGPNTM.DataConstruct import LoadDynamicHeteGraph
from src.sota.FOREST.model import RNNModel
from src.sota.NDM.transformer.Models import Decoder
from src.sota.HiDAN.model import HiDANModel
from src.sota.HiDAN.config import Config
from src.sota.MSHGAT.model import MSHGAT
from src.sota.MSHGAT.graphConstruct import ConRelationGraph, ConHyperGraphList
from src.sota.MSHGAT.dataLoader import Split_data
from src.sota.MINDS.Module import MINDS
from src.sota.MINDS.HypergraphUtil import DynamicCasHypergraph
from src.sota.InfVAE import InfVAE
import numpy as np
import argparse
import shutil
import time
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dhg

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_printoptions(threshold=10_000)

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--dataset', type=str, default='Twitter-Huangxin', help="available options are ['Weibo-Aminer','Twitter-Huangxin']")
parser.add_argument('--model', type=str, default='heteredgegat', help="available options are ['densegat','heteredgegat','diffusiongat','dhgpntm','semantic','semanticgat','gthnn']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
# >> Preprocess
parser.add_argument('--min-user-participate', type=int, default=2, help="Min User Participate in One Cascade")
parser.add_argument('--max-user-participate', type=int, default=500, help="Max User Participate in One Cascade")
# parser.add_argument('--train-ratio', type=float, default=0.8, help="Training ratio (0, 1)")
# parser.add_argument('--valid-ratio', type=float, default=0.1, help="Validation ratio (0, 1)")
parser.add_argument('--tmax', type=int, default=120, help="Max Time in the Observation Window")
parser.add_argument('--n-interval', type=int, default=40, help="Number of Time Intervals in the Observation Window")
parser.add_argument('--n-component', type=int, default=None, help="Number of Prominent Component Topic Classes Foreach Topic")
# >> Model
parser.add_argument('--graph-filename', type=str, default="full", help="")
parser.add_argument('--window-size', type=int, default=7, help="Window Size of Building Topical Edges")
parser.add_argument('--instance-normalization', action='store_true', default=False, help="Enable instance normalization")
parser.add_argument('--use-gat', type=int, default=1, help="Use GAT as Backbone")
parser.add_argument('--use-time-decay', type=int, default=1, help="Use Time Embedding")
# parser.add_argument('--use-topic-selection', type=int, default=1, help="")
parser.add_argument('--use-motif', action='store_true', default=False, help="Use Motif-Enhanced Graph")
parser.add_argument('--use-add-attn', action='store_true', default=False, help="")
# parser.add_argument('--use-topic-preference', action='store_true', default=False, help="Use Hand-crafted Topic Preference Weights to Aggregate topic-enhanced graph embeds")
# parser.add_argument('--use-tweet-feat', action='store_true', default=False, help="Use Tweet-Side Feat Aggregated From Tag Embeddings")
# parser.add_argument('--unified-dim', type=int, default=128, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--d_model', type=int, default=64, help='Options in ScheduledOptim')
parser.add_argument('--n_warmup_steps', type=int, default=1000, help='Options in ScheduledOptim')
parser.add_argument('--patience', type=int, default=10, help='Patience Steps of EarlyStopping')
# >> Graph Denoising
# parser.add_argument('--type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
# parser.add_argument('--k', type=int, default=30)
# parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
# parser.add_argument('--gamma', type=float, default=0.9)
# parser.add_argument('--activation_learner', type=str, default='relu', choices=["relu", "tanh"])
# parser.add_argument('--sparse', type=int, default=0)
# parser.add_argument('--use-diffusion-graph', action='store_true', default=True, help="Use Diffusion Graph")
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, help='Number of epochs to train.')
# [default] hetersparsegat: 3e-3, hypergat: 3e-3, densegat: 3e-2
parser.add_argument('--lr', type=float, default=3e-2, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="4", help="Heads in each layer, splitted with comma")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--gpu', type=str, default="cuda:0", help="Select GPU")
parser.add_argument('--graph-topk', type=int, default=20, help="")
# >> Ablation Study
parser.add_argument('--use-random-multiedge', type=int, default=0, help="Use Random Multi-Edge to build Heter-Edge-Matrix if set true (Available only when model='heteredgegat')")
parser.add_argument('--use-multi-deepwalk-feat', action='store_true', default=False, help="Use Multi-Heter Deepwalk-Feature if set true (Available only when model='heteredgegat')")
# parser.add_argument('--use-adj', type=int, default=1, help="Use Adj Matrix to Mask Attn if set true (Available only when model='heteredgegat')")
# >> Comparison(New)
parser.add_argument('--tweet2vec', type=int, default=0, help="Utilize texts as feat vectors (No rel. with whether to use textual diffusion channels)")
parser.add_argument('--tweet2graph', type=int, default=1, help="Utilize texts as textual graphs")
parser.add_argument('--use-random-vec', type=int, default=1, help="Use Random Vectors (Otherwise use User-Feats or User+Tweet-Feats)")
parser.add_argument('--semantic-feat-file', type=str, default=None, help="Optional semantic feature file relative to dataset dir; defaults to the LLM user interest embedding file")
parser.add_argument('--sparsity', type=int, default=100, help='Sparsity Comparison (0-100)')
parser.add_argument('--tw-version', type=int, default=4, help="Topic graph version/threshold suffix: 4 -> tw4/wb4, 5 -> tw5/wb5, 7 -> tw7/wb7")
parser.add_argument('--infvae-layer-config', type=str, default="256,128,64", help="Inf-VAE graph encoder layer config, matching the released flags")
parser.add_argument('--infvae-hidden-dim', type=int, default=128, help="Deprecated; use --infvae-layer-config instead")
parser.add_argument('--infvae-latent-dim', type=int, default=64, help="Latent social/temporal dimension of Inf-VAE")
parser.add_argument('--infvae-vae-weight', type=float, default=0.1, help="Weight of the Inf-VAE graph VAE/MAP regularizer")
parser.add_argument('--infvae-neg-samples', type=int, default=4096, help="Number of graph edges sampled for Inf-VAE reconstruction loss")
parser.add_argument('--infvae-lambda-s', type=float, default=1.0, help="Inf-VAE sender prior strength")
parser.add_argument('--infvae-lambda-r', type=float, default=0.01, help="Inf-VAE receiver prior strength")
parser.add_argument('--infvae-lambda-p', type=float, default=0.1, help="Inf-VAE popularity prior strength")
parser.add_argument('--infvae-pos-weight', type=float, default=1.0, help="Positive edge weight for Inf-VAE graph reconstruction")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.use_gat = args.use_gat == 1
args.use_time_decay = args.use_time_decay == 1
# args.use_adj = args.use_adj == 1
# args.use_k_adj = args.use_adj == 2
# args.use_topic_selection = args.n_component is not None
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def get_cascade_criterion(user_size):
    ''' With PAD token zero weight '''
    weight = torch.ones(user_size)
    weight[PAD] = 0
    weight[EOS] = 0
    return torch.nn.CrossEntropyLoss(weight)

def get_performance(criterion, pred_cascade, gold_cascade):
    '''
    pred_cascade: (#samples, #user_size), gold_cascade: (#samples,)
    '''
    gold_cascade = gold_cascade.contiguous().view(-1)
    pred_cascade[pred_cascade == -float('inf')] = 0
    pred_cascade = pred_cascade.view(gold_cascade.size(0), -1)
    loss = criterion(pred_cascade, gold_cascade)

    pred_cascade = pred_cascade.max(1)[1]
    n_correct = pred_cascade.data.eq(gold_cascade.data)
    n_correct = n_correct.masked_select((gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data).sum().float()
    return loss, n_correct

def get_performance2(opt, cascade_crit, pred_cascade, gold_cascade):  
    gold_cascade = gold_cascade.contiguous().view(-1)
    inte_sta = np.array([0]*opt.num_heads)
    pred_cascade = pred_cascade.view(gold_cascade.size(0),opt.num_heads,opt.user_size).max(1)[0]
    pred_cascade[pred_cascade == -float('inf')] = 0
    cascade_loss = cascade_crit(pred_cascade, gold_cascade)
    # gold_cate = torch.from_numpy(np.array(range(opt.num_heads))).unsqueeze(-1).repeat(1,opt.user_size).view(-1).to(opt.device)
    # regular_loss = regular_crit(regular_outputs,gold_cate)
    #cascade prediction
    pred_cascade = pred_cascade.max(1)[1]
    gold_cascade = gold_cascade.contiguous().view(-1)
    n_cascade_correct = pred_cascade.data.eq(gold_cascade.data)
    n_cascade_correct = n_cascade_correct.masked_select((gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data).sum().float()
    return cascade_loss, None, n_cascade_correct,inte_sta

def get_scores(pred_cascade:torch.Tensor, gold_cascade:torch.Tensor, k_list=[10,50,100]):
    gold_cascade = gold_cascade.contiguous().view(-1)           # (#samples,)
    pred_cascade = pred_cascade.view(gold_cascade.size(0), -1)  # (#samples, #user_size)
    pred_cascade = pred_cascade.detach().cpu().numpy()
    gold_cascade = gold_cascade.detach().cpu().numpy()
    scores = compute_metrics(pred_cascade, gold_cascade, k_list)
    return scores

def get_scores2(opt, pred_cascade:torch.Tensor, gold_cascade:torch.Tensor, k_list=[10,50,100]):
    gold_cascade = gold_cascade.contiguous().view(-1)           # (#samples,)
    user_num,user_size = pred_cascade.size(0), int(pred_cascade.size(1)/opt.num_heads)
    pred_cascade = pred_cascade.view(user_num,opt.num_heads,user_size).max(1)[0]
    pred_cascade = pred_cascade.detach().cpu().numpy()
    gold_cascade = gold_cascade.detach().cpu().numpy()
    # scores, _ = portfolio(pred_cascade, gold_cascade, k_list)
    scores = compute_metrics(pred_cascade, gold_cascade, k_list)
    return scores


def build_gthnn_hypergraphs(cascades, user_size, num_interval):
    hypergraphs = []
    for period_i in range(num_interval):
        node_ids = []
        hyperedge_ids = []
        hedge_id = 0
        for cascade in cascades:
            buckets = {}
            for user, interval in zip(cascade['user'], cascade['interval']):
                user = int(user)
                if user == PAD or user == EOS or user < 0 or user >= user_size:
                    continue
                interval = max(0, min(num_interval - 1, int(interval)))
                time_bin = num_interval - 1 - interval
                if time_bin == period_i:
                    buckets.setdefault(hedge_id, []).append(user)
            for users in buckets.values():
                unique_users = list(dict.fromkeys(users))
                if len(unique_users) < 2:
                    continue
                node_ids.extend(unique_users)
                hyperedge_ids.extend([hedge_id] * len(unique_users))
                hedge_id += 1

        if len(node_ids) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.LongTensor([node_ids, hyperedge_ids])
        hypergraphs.append(edge_index)
        logger.info("GTHNN hypergraph period=%s incidence_edges=%s hyperedges=%s", period_i, len(node_ids), hedge_id)
    return hypergraphs

def MAE(y, y_predicted):
    y_predicted = y_predicted.squeeze()
    mae = torch.abs(y_predicted - y)
    # sum_sq_error = torch.sum(sq_error)
    # mse = sum_sq_error / label.size()
    mae = torch.mean(mae)
    return mae

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    # if seq.is_cuda:
    #     PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    # if seq.is_cuda:
    #     ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    # print("masked_seq ",masked_seq.size())
    return masked_seq

def train(epoch_i, data, graph, model, optimizer, loss_func, writer, user_size, log_desc='train_'):
    model.train()

    # if args.model == 'heteredgegat' and not args.use_diffusion_graph:
    #     data['graph_learner'].train()
    #     data['optimizer_learner'].zero_grad()
    
    loss, correct, total = 0., 0., 0.
    for _, batch in enumerate(data['batch']):
    # for _, batch in enumerate(tqdm(data['batch'])):

        cas_users, cas_tss, cas_intervals, cas_classids, cas_idx, tgt_len = batch
        # print("cas_classids", cas_classids)
        # print("cas_idx", cas_idx)
        if args.cuda:
            cas_users = cas_users.to(args.gpu)
            cas_tss = cas_tss.to(args.gpu)
            cas_intervals = cas_intervals.to(args.gpu)
            cas_idx = cas_idx.to(args.gpu)
            tgt_len = tgt_len.to(args.gpu)
        gold_cascade = cas_users[:, 1:]
        # logger.info(f"gold_cascade={gold_cascade}")
        
        optimizer.zero_grad()
        if args.model == 'densegat':
            pred_cascade = model(cas_users, cas_intervals, graph)
        elif args.model == 'mshgat':
            pred_cascade = model(cas_users, cas_tss, cas_idx, graph, data['hypergraph_list'])
        elif args.model == 'minds':
            pred_micro, pred_macro, loss_adv, loss_diff = model(data['hypergraph_list'], graph, cas_users)
        elif args.model == 'heteredgegat':
            # if args.use_diffusion_graph:
            #     learned_adj = data['diffusion_graph']
            # else:
            #     learned_adj = data['graph_learner'](data['features'])
            #     if args.type_learner == 'fgp':
            #         learned_adj[learned_adj<1e-2] = 0.
            #     edge_index, edge_weight = dense_to_sparse(learned_adj)
            #     learned_adj = Data(edge_index=edge_index, edge_weight=edge_weight)
            # pred_cascade = model(cas_users, cas_intervals, cas_classids, data['hedge_graphs'], multi_deepwalk_feat=data['multi_deepwalk_feat'])
            pred_cascade = model(cas_users, cas_intervals, cas_classids, data['hedge_graphs'], feats=data['feat'], multi_deepwalk_feat=data['multi_deepwalk_feat'])
        elif args.model == 'gthnn':
            pred_cascade = model(cas_users, cas_intervals, graph, data['gthnn_hypergraphs'])
        elif args.model == 'semantic':
            pred_cascade = model(cas_users, cas_intervals, data['semantic_feat'])
        elif args.model == 'semanticgat':
            pred_cascade = model(cas_users, cas_intervals, graph, data['semantic_feat'])
        # elif args.model == 'diffusiongat':
        #     pred_cascade = model(cas_users, cas_intervals, data['diffusion_graph'])
        elif args.model == 'tan':
            pred_cascade, _ = model((cas_users, cas_intervals, None, None))
        elif args.model == 'dhgpntm':
            pred_cascade = model(cas_users, cas_tss, cas_intervals, None, data['diffusion_graph'])
        elif args.model == 'forest':
            pred_cascade, _ = model(cas_users)
        elif args.model == 'ndm':
            pred_cascade = model(cas_users)
        elif args.model == 'hidan':
            pred_cascade = model(cas_users, cas_intervals)
        elif args.model == 'infvae':
            pred_cascade = model(cas_users, cas_intervals, graph)

        # scores_batch = get_scores(pred_cascade, gold_cascade, [10])
        
        if args.model == 'tan':
            loss_batch, _, n_correct, _ = get_performance2(data['opt'], loss_func, pred_cascade, gold_cascade)
        elif args.model == 'minds':
            lambda_loss = 0.3
            gamma_loss = 0.05
            # gold = cas_users[:, 1:]
            mask = get_previous_user_mask(cas_users[:, :-1].cpu(), user_size).to(args.gpu)
            micro_loss, n_correct = get_performance(loss_func,(pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1)), gold_cascade)
            # print(f"pred_macro shape: {pred_macro.shape}")
            # print(f"tgt_len shape: {tgt_len.shape}")
            pred_macro = pred_macro.squeeze(-1)
            if pred_macro.shape[0] != tgt_len.shape[0]:
                pred_macro = pred_macro[:tgt_len.shape[0]]
            macro_loss = MAE(tgt_len, pred_macro)
            loss_batch = (1 - lambda_loss) * micro_loss + lambda_loss * macro_loss
            loss_batch += loss_adv
            loss_batch += gamma_loss * loss_diff
        elif args.model == 'hidan':
            gold_user = cas_users[:, 1]                
            loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_user)
        elif args.model == 'infvae':
            loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
            loss_batch = loss_batch + args.infvae_vae_weight * model.regularization_loss(graph)
        else:
            loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
        loss_batch.backward()
        optimizer.step()
        optimizer.update_learning_rate()

        n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
        loss += loss_batch.item()
        # logger.info(f"loss={loss}")
        correct += n_correct.item()
        total += n_words

        # 及时释放，断开计算图
        # del pred_micro, pred_macro, mask, micro_loss, macro_loss, loss_adv, loss_diff
        del pred_cascade
    
    # if args.model == 'heteredgegat' and not args.use_diffusion_graph:
    #     data['optimizer_learner'].step()
    
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1)
    writer.add_scalar(log_desc+'acc',  correct/total, epoch_i+1)
    torch.cuda.empty_cache() # 清理缓存，让 cuda reserved 数字回落
    return loss/total, correct/total

def evaluate(epoch_i, data, graph, model, optimizer, loss_func, writer, user_size, k_list=[10,50,100], log_desc='valid_', return_loss_acc=False):
    model.eval()

    # if args.model == 'heteredgegat' and not args.use_diffusion_graph:
    #     data['graph_learner'].eval()
    
    loss, correct, total = 0., 0., 0.
    scores = {'MRR': 0,}
    for k in k_list:
        scores[f'hits@{k}'] = 0
        scores[f'map@{k}'] = 0
    
    with torch.no_grad():
        for _, batch in enumerate((data['batch'])):
        # for _, batch in enumerate(tqdm(data['batch'])):

            cas_users, cas_tss, cas_intervals, cas_classids, cas_idx, tgt_len = batch
            if args.cuda:
                cas_users = cas_users.to(args.gpu)
                cas_tss = cas_tss.to(args.gpu)
                cas_intervals = cas_intervals.to(args.gpu)
                cas_idx = cas_idx.to(args.gpu)
                tgt_len = tgt_len.to(args.gpu)
            gold_cascade = cas_users[:, 1:]
            
            optimizer.zero_grad()
            if args.model == 'densegat':
                pred_cascade = model(cas_users, cas_intervals, graph)
            elif args.model == 'mshgat':
                pred_cascade = model(cas_users, cas_tss, cas_idx, graph, data['hypergraph_list'])
            elif args.model == 'minds':
                pred_micro, pred_macro, loss_adv, loss_diff = model(data['hypergraph_list'], graph, cas_users)
            elif args.model == 'heteredgegat':
                # if args.use_diffusion_graph:
                #     learned_adj = data['diffusion_graph']
                # else:
                #     learned_adj = data['graph_learner'](data['features'])
                #     if args.type_learner == 'fgp':
                #         learned_adj[abs(learned_adj-1)>1e-6] = 0
                #     edge_index, edge_weight = dense_to_sparse(learned_adj)
                #     learned_adj = Data(edge_index=edge_index, edge_weight=edge_weight)
                # pred_cascade = model(cas_users, cas_intervals, cas_classids, data['hedge_graphs'], multi_deepwalk_feat=data['multi_deepwalk_feat'])
                pred_cascade = model(cas_users, cas_intervals, cas_classids, data['hedge_graphs'], feats=data['feat'], multi_deepwalk_feat=data['multi_deepwalk_feat'])
            elif args.model == 'gthnn':
                pred_cascade = model(cas_users, cas_intervals, graph, data['gthnn_hypergraphs'])
            elif args.model == 'semantic':
                pred_cascade = model(cas_users, cas_intervals, data['semantic_feat'])
            elif args.model == 'semanticgat':
                pred_cascade = model(cas_users, cas_intervals, graph, data['semantic_feat'])
            # elif args.model == 'diffusiongat':
            #     pred_cascade = model(cas_users, cas_intervals, data['diffusion_graph'])
            elif args.model == 'tan':
                pred_cascade, _ = model((cas_users, cas_intervals, None, None))
            elif args.model == 'dhgpntm':
                pred_cascade = model(cas_users, cas_tss, cas_intervals, None, data['diffusion_graph'])
            elif args.model == 'forest':
                pred_cascade, _ = model(cas_users)
            elif args.model == 'ndm':
                pred_cascade = model(cas_users)
                pred_cascade = pred_cascade[0]
            elif args.model == 'hidan':
                pred_cascade = model(cas_users, cas_intervals)
            elif args.model == 'infvae':
                pred_cascade = model(cas_users, cas_intervals, graph)
            
            if args.model == 'tan':
                loss_batch, _, n_correct, _ = get_performance2(data['opt'], loss_func, pred_cascade, gold_cascade)
            elif args.model == 'minds':
                lambda_loss = 0.3
                gamma_loss = 0.05
                # gold = cas_users[:, 1:]
                mask = get_previous_user_mask(cas_users[:, :-1].cpu(), user_size).to(args.gpu)
                micro_loss, n_correct = get_performance(loss_func,(pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1)), gold_cascade)
                macro_loss = MAE(tgt_len, pred_macro)
                loss_batch = (1 - lambda_loss) * micro_loss + lambda_loss * macro_loss
                loss_batch += loss_adv
                loss_batch += gamma_loss * loss_diff
            elif args.model == 'hidan':
                gold_user = cas_users[:, 1]                
                loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_user)
            elif args.model == 'infvae':
                loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
                loss_batch = loss_batch + args.infvae_vae_weight * model.regularization_loss(graph)
            else:
                loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
                    
            n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
            loss += loss_batch.item()
            correct += n_correct.item()
            total += n_words

            if args.model == 'tan':
                scores_batch = get_scores2(data['opt'], pred_cascade, gold_cascade, k_list)
            elif args.model == 'minds':
                mask = get_previous_user_mask(cas_users[:, :-1].cpu(), user_size).to(args.gpu)
                y_pred = (pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1)).to(args.gpu)
                # y_pred = y_pred.detach().cpu().numpy()
                scores_batch = get_scores(y_pred, gold_cascade, k_list)
            elif args.model == 'hidan':
                B, L = cas_users.size()
                all_logits = []
                all_labels = []
                for t in range(1, L):
                    prefix     = cas_users[:, :t]
                    time_pref  = cas_intervals[:, :t]
                    logits_t   = model(prefix, time_pref)
                    all_logits.append(logits_t)
                    all_labels.append(cas_users[:, t])
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                valid_mask = (all_labels.ne(PAD) & all_labels.ne(EOS))
                valid_logits = all_logits[valid_mask]      # [n_valid, user_size]
                valid_labels = all_labels[valid_mask]
                scores_batch = get_scores(valid_logits, valid_labels, k_list)
            elif args.model == 'infvae':
                scores_batch = get_scores(pred_cascade, gold_cascade, k_list)
            else:
                scores_batch = get_scores(pred_cascade, gold_cascade, k_list)
            
            # scores['prec'] += scores_batch['prec'] * n_words
            # scores['rec'] += scores_batch['rec'] * n_words
            # scores['F1'] += scores_batch['F1'] * n_words
            scores['MRR'] += scores_batch['MRR'] * n_words
            for k in k_list:
                scores[f'hits@{k}'] += scores_batch[f'hits@{k}'] * n_words
                scores[f'map@{k}'] += scores_batch[f'map@{k}'] * n_words
    
    model.train()
    # if args.model == 'heteredgegat' and not args.use_diffusion_graph:
    #     data['graph_learner'].train()
    
    # scores['prec'] /= total
    # scores['rec'] /= total
    # scores['F1'] /= total
    scores['MRR'] /= total
    for k in k_list:
        scores[f'hits@{k}'] /= total
        scores[f'map@{k}'] /= total
    # logger.info(f"MRR={scores['MRR']}, hits@10={scores['hits@10']}, map@10={scores['map@10']}, hits@50={scores['hits@50']}, map@100={scores['map@100']}, hits@100={scores['hits@100']}, map@100={scores['map@100']},")
    
    loss_avg = loss/total
    acc_avg = correct/total
    writer.add_scalar(log_desc+'loss', loss_avg, epoch_i+1); writer.add_scalar(log_desc+'acc', acc_avg, epoch_i+1); writer.add_scalar(log_desc+'mrr', scores['MRR'], epoch_i+1)
    writer.add_scalar(log_desc+'hits@10',  scores['hits@10'],  epoch_i+1);  writer.add_scalar(log_desc+'map@10',  scores['map@10'],  epoch_i+1)
    writer.add_scalar(log_desc+'hits@50',  scores['hits@50'],  epoch_i+1);  writer.add_scalar(log_desc+'map@50',  scores['map@50'],  epoch_i+1)
    writer.add_scalar(log_desc+'hits@100', scores['hits@100'], epoch_i+1);  writer.add_scalar(log_desc+'map@100', scores['map@100'], epoch_i+1)
    # writer.add_scalar(log_desc+'auc', auc, epoch_i+1);                    writer.add_scalar(log_desc+'f1', f1, epoch_i+1)
    # writer.add_scalar(log_desc+'prec', prec, epoch_i+1);                  writer.add_scalar(log_desc+'rec', rec, epoch_i+1)
    torch.cuda.empty_cache()
    if return_loss_acc:
        return scores, loss_avg, acc_avg
    return scores

def expand_(feat:torch.Tensor, shape:tuple, pos=0)->torch.Tensor:
    shape_ = feat.size()
    # assert len(shape_) == len(shape)
    for idx, (dim_, dim) in enumerate(zip(shape_, shape)):
        if dim_ == dim: continue
        # if dim_ > dim: raise Exception
        size_ = list(shape); size_[idx] = dim-dim_
        # TODO: pos=0
        feat = torch.cat((torch.zeros(size=size_).to(feat.device), feat), dim=idx)
    return feat

def main():
    # torch.set_num_threads(4)

    dataset_dirpath = os.path.join(DATA_ROOTPATH, args.dataset)
    if args.dataset == 'Twitter-Huangxin':
        sub10000_dirpath = os.path.join(dataset_dirpath, "sub10000")
        if os.path.isdir(sub10000_dirpath):
            dataset_dirpath = sub10000_dirpath
        # dataset_dirpath += '/sub5000'
    
    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    # NOTE: set `load_dict=True` for all datasets
    train_data = DataConstruct(dataset_dirpath=dataset_dirpath, batch_size=args.batch_size, seed=args.seed, 
                               tmax=args.tmax, num_interval=args.n_interval, 
                               n_component=args.n_component, data_type=0, sparsity=args.sparsity, load_dict=True)
    valid_data = DataConstruct(dataset_dirpath=dataset_dirpath, batch_size=args.batch_size, seed=args.seed, 
                               tmax=args.tmax, num_interval=args.n_interval, 
                               n_component=args.n_component, data_type=1, sparsity=args.sparsity, load_dict=True)
    test_data  = DataConstruct(dataset_dirpath=dataset_dirpath, batch_size=args.batch_size, seed=args.seed, 
                               tmax=args.tmax, num_interval=args.n_interval, 
                               n_component=args.n_component, data_type=2, sparsity=args.sparsity, load_dict=True)

    train_d = {'batch': train_data}; valid_d = {'batch': valid_data}; test_d = {'batch': test_data}

    # user_ids = read_user_ids(f"{dataset_dirpath}/train_withcontent.data", f"{dataset_dirpath}/valid_withcontent.data", f"{dataset_dirpath}/test_withcontent.data")
    # edges = get_static_subnetwork(user_ids)
    # _, edges = reindex_edges(user_ids, edges)
    if args.model == 'minds':
        user_edges = load_pickle(os.path.join(dataset_dirpath, "edges.data"))
        graph = dhg.Graph(train_data.user_size, user_edges, device=args.gpu)
        logger.info(f'#Link: {len(graph.e[0])}')
    else:
        user_edges = load_pickle(os.path.join(dataset_dirpath, "edges.data"))
        user_edges = user_edges + [(i,i) for i in range(train_data.user_size)]
        user_edges = list(zip(*user_edges))
        edges_t = torch.LongTensor(user_edges) # (2,#num_edges)
        weight_t = torch.FloatTensor([1]*edges_t.size(1))
        graph = Data(edge_index=edges_t, edge_weight=weight_t)
    # if args.cuda:
    #     graph = graph.to(args.gpu)
    
    # Use Manual Feats or Random Feats
    new_d = {'feat': None}
    if args.use_random_vec == 0:
        vertex_feat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/vertex_feature_user{train_data.user_size}.npy"))
        three_sort_feat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/three_sort_feature_user{train_data.user_size}.npy"))
        deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/deepwalk_emb_user{train_data.user_size}.data"), max_idx=train_data.user_size-1)
        user_side_emb = torch.cat([torch.FloatTensor(vertex_feat),torch.FloatTensor(deepwalk_feat),torch.FloatTensor(three_sort_feat),],dim=1)
        if args.cuda:
            user_side_emb = user_side_emb.to(args.gpu)
        
        if args.tweet2vec:
            tweet_aggy_feat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/llm/tag_embs_aggbyuser_model_xlm-roberta-base_pca_dim128_user{train_data.user_size}.pkl"))
            tweet_side_emb = torch.FloatTensor(tweet_aggy_feat)
            # tweet_side_emb = expand_(tweet_side_emb, shape=(user_side_emb.size(0),tweet_side_emb.size(1)))
            if args.cuda:
                tweet_side_emb = tweet_side_emb.to(args.gpu)
            
            user_side_emb = torch.cat([user_side_emb, tweet_side_emb], dim=1)
        
        new_d = {'feat': user_side_emb}
    train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

    if args.model in ('semantic', 'semanticgat'):
        semantic_feat_file = args.semantic_feat_file
        if semantic_feat_file is None:
            semantic_feat_file = f"llm/tag_embs_aggbyuser_model_xlm-roberta-base_pca_dim128_user{train_data.user_size}.pkl"
        semantic_feat_path = os.path.join(dataset_dirpath, semantic_feat_file)
        if not os.path.isfile(semantic_feat_path):
            semantic_feat_path = os.path.join(DATA_ROOTPATH, args.dataset, semantic_feat_file)
        if not os.path.isfile(semantic_feat_path):
            raise FileNotFoundError(f"Semantic feature file not found: {semantic_feat_file}")
        semantic_feat = torch.FloatTensor(load_pickle(semantic_feat_path))
        if semantic_feat.size(0) > train_data.user_size:
            semantic_feat = semantic_feat[:train_data.user_size]
        semantic_feat = expand_(semantic_feat, shape=(train_data.user_size, semantic_feat.size(1)))
        if args.cuda:
            semantic_feat = semantic_feat.to(args.gpu)
        logger.info("semantic feature loaded from %s, shape=%s", semantic_feat_path, tuple(semantic_feat.size()))
        new_d = {'semantic_feat': semantic_feat}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

    if args.model == 'semantic':
        hidden_dim = n_units[-1] * n_heads[-1]
        model = SemanticSequenceNetwork(
            user_size=train_data.user_size,
            semantic_feat_dim=train_d['semantic_feat'].size(1),
            hidden_dim=hidden_dim,
            num_interval=args.n_interval,
            dropout=args.dropout,
        )

    elif args.model == 'semanticgat':
        if args.cuda:
            graph = graph.to(args.gpu)
        n_feat = n_units[0] * n_heads[0] if args.use_gat else n_units[0]
        model = SemanticGATNetwork(
            user_size=train_data.user_size,
            semantic_feat_dim=train_d['semantic_feat'].size(1),
            n_feat=n_feat,
            n_units=n_units,
            n_heads=n_heads,
            num_interval=args.n_interval,
            attn_dropout=args.attn_dropout,
            dropout=args.dropout,
            use_gat=args.use_gat,
        )

    elif args.model == 'gthnn':
        hidden_dim = n_units[-1] * n_heads[-1]
        gthnn_hypergraphs = build_gthnn_hypergraphs(train_data._train_data, train_data.user_size, args.n_interval)
        if args.cuda:
            gthnn_hypergraphs = [hgraph.to(args.gpu) for hgraph in gthnn_hypergraphs]
        new_d = {'gthnn_hypergraphs': gthnn_hypergraphs}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)
        model = GTHNNNetwork(
            user_size=train_data.user_size,
            hidden_dim=hidden_dim,
            num_interval=args.n_interval,
            n_heads=n_heads[-1],
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
        )

    elif args.model == 'densegat':
        n_feat = n_units[0]*n_heads[0] if args.use_gat else n_units[0]
        model = BasicGATNetwork(n_feat=n_feat, n_units=n_units, n_heads=n_heads, num_interval=args.n_interval, shape_ret=(n_feat,train_data.user_size), 
            attn_dropout=args.attn_dropout, dropout=args.dropout, use_gat=args.use_gat)

    elif args.model == 'mshgat':
        total_cascades, timestamps = Split_data(dataset_dirpath, load_dict=True)
        logger.info("len of total_cascades: %s", len(total_cascades))
        logger.info("len of timestamps: %s", len(timestamps))
        hypergraph_list = ConHyperGraphList(cascades=total_cascades, timestamps=timestamps, user_size=train_data.user_size)
        # 修改3, 稀疏化 graph_list（节省显存）
        # graph_list, root_list = hypergraph_list
        # for t in graph_list:
        #     if not graph_list[t].is_sparse:
        #         graph_list[t] = graph_list[t].to_sparse().coalesce()
        #     else:
        #         graph_list[t] = graph_list[t].coalesce()
        # hypergraph_list = (graph_list, root_list)
        # print("\n=== 检查 graph_list ===")
        # print(f"graph_list 类型: {type(graph_list)}") 
        # print(f"graph_list 包含的时间戳数量: {len(graph_list)}")
        # # 遍历并打印每个子图的形状
        # for ts, sub_graph in graph_list.items():
        #     print(f"时间戳 {ts}: sub_graph.shape = {sub_graph.shape}")

        save_pickle(hypergraph_list, os.path.join(dataset_dirpath, "hypergraph_list.data"))
        if args.cuda:
            dict_part = {k: v.to(args.gpu) for k, v in hypergraph_list[0].items()}
            tensor_part = hypergraph_list[1].to(args.gpu)
            hypergraph_list = (dict_part, tensor_part)
        
        new_d = {'hypergraph_list':hypergraph_list}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)
        model = MSHGAT(user_size = train_data.user_size, dropout = 0.1, device=args.gpu)

    elif args.model == 'minds':
        total_cascades, timestamps = Split_data(dataset_dirpath, load_dict=True)
        # relation_graph = RelationGraph(dataset, device)
        hypergraph_list = DynamicCasHypergraph(examples=total_cascades, examples_times=timestamps, user_size=train_data.user_size, device=args.gpu, step_split=8)
        new_d = {'hypergraph_list':hypergraph_list}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)
        model = MINDS(user_size=train_data.user_size, embed_dim=64, step_split=8, max_seq_len=500, task_num=2, device=args.gpu)

    elif args.model == 'heteredgegat':
        graph_prefix = "wb" if args.dataset == "Weibo-Aminer" else "tw"
        base_filename = f"{graph_prefix}{args.tw_version}_new_topic_diffusion_graph"
        if args.use_random_multiedge:
            base_filename += "_random"
            logger.info("use_random_multiedge")
        elif args.use_motif:
            base_filename = "topic_diffusion_motif_graph"
        else:
            # base_filename += "_full"
            base_filename += "_{}".format(args.graph_filename)
        
        sparsity_suffix = ""
        if args.sparsity < 100:
            sparsity_suffix = f"_sparsity{args.sparsity}.0"
        
        classid2simmat = load_pickle(os.path.join(dataset_dirpath, f"topic_llm_ppx/{base_filename}_windowsize{args.window_size}{sparsity_suffix}.data"))
        # classid2simmat = load_pickle(os.path.join(dataset_dirpath, f"topic_llm2/{base_filename}_windowsize{args.window_size}{sparsity_suffix}.data"))
        # if args.cuda:
        #     classid2simmat = {classid:simmat.to(args.gpu) for classid, simmat in classid2simmat.items()}
        
        # n_adj = max(classid2simmat.keys())+1
        
        topk = args.graph_topk
        n_adj = min(len(classid2simmat), topk)
        logger.info("n_adj : {}".format(n_adj))
        if args.tweet2graph == 0:
            # hedge_graphs = [graph] * (n_adj+1)
            if args.cuda:
                graph = graph.to(args.gpu)
            hedge_graphs = [graph] * n_adj
        else:
            # hedge_graphs = [classid2simmat[classid] if classid in classid2simmat else graph for classid in range(n_adj)]
            
            # select topk interest graphs
            clasid2len = {k: v.edge_index.size(1) for k,v in classid2simmat.items()}
            for k in sorted(clasid2len, key=clasid2len.get, reverse=True)[topk:]:
                classid2simmat.pop(k)
            for k,v in classid2simmat.items():
                logger.info("k = {}, v = {}".format(k, v.edge_index.size(1)))
            hedge_graphs = [graph for _, graph in classid2simmat.items()]
            if args.cuda:
                hedge_graphs = [graph.to(args.gpu) for graph in hedge_graphs]
        
        """
        clasid2len = {k: v.edge_index.size(1) for k, v in classid2simmat.items()}
        edge_counts = torch.tensor(list(clasid2len.values()), dtype=torch.float32, device=args.gpu)

        num_topics = len(clasid2len)
        topic_logits = torch.nn.Parameter(0.1 * edge_counts + torch.randn(num_topics, device=args.gpu) * 0.01)

        optimizer = torch.optim.Adam([topic_logits], lr=0.01)

        for _ in range(100): 
            optimizer.zero_grad()
            
            selection_probs = torch.softmax(topic_logits, dim=0)
            selected_topics = torch.bernoulli(selection_probs)

            edge_loss = -torch.sum(selected_topics * edge_counts)  
            num_selected_loss = torch.sum(selected_topics)  

            alpha = edge_loss.abs().mean() / num_selected_loss.abs().mean()
            loss = edge_loss + alpha * num_selected_loss

            loss.backward()
            optimizer.step()

        topk = args.graph_topk
        final_selection = topic_logits > torch.topk(topic_logits, k=topk).values[-1]

        selected_classes = [k for i, k in enumerate(clasid2len.keys()) if final_selection[i]]
        classid2simmat = {k: classid2simmat[k] for k in selected_classes}
        hedge_graphs = [graph for _, graph in classid2simmat.items()]
        n_adj = len(hedge_graphs)

        logger.info("n_adj : {}".format(n_adj)) 
        for k, v in classid2simmat.items():
            logger.info("k = {}, v = {}".format(k, v.edge_index.size(1))) 

        if args.cuda:
            hedge_graphs = [graph.to(args.gpu) for graph in hedge_graphs]
        """
        # diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_graph.data"))
        # if args.cuda:
        #     diffusion_graph = diffusion_graph[sorted(diffusion_graph.keys())[-1]].to(args.gpu)
        
        # user_topic_preference = None
        # if args.use_topic_preference:
        #     user_topic_preference = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/llm/user_topic_pref_cnt.pkl"))
        #     # -1 use mean vector
        #     user_topic_preference = torch.cat((user_topic_preference, torch.mean(user_topic_preference, dim=1).view(-1,1)), dim=1)
        #     if args.cuda:
        #         user_topic_preference = user_topic_preference.to(args.gpu)

        n_feat = n_units[0] * n_heads[0]
        n_units = [nu * nh for nu, nh in zip(n_units, n_heads)] if not args.use_gat else n_units
        use_add_attn = args.use_add_attn
        use_topic_selection = args.n_component is not None
        random_feat_dim = train_d['feat'].size(1) if train_d['feat'] is not None else None
        # logger.info("random_feat_dim: {}".format(random_feat_dim))
        model = HeterEdgeGATNetwork(user_size=train_data.user_size, n_feat=n_feat, n_adj=n_adj, num_interval=args.n_interval, n_comp=args.n_component, 
            n_units=n_units, n_heads=n_heads, attn_dropout=args.attn_dropout, dropout=args.dropout, 
            use_gat=args.use_gat, use_time_decay=args.use_time_decay, 
            use_add_attn=use_add_attn, use_topic_selection=use_topic_selection, random_feat_dim=random_feat_dim)
        
        # Graph Denoising
        # features = torch.cat([torch.FloatTensor(tweet_aggy_feat),],dim=1)
        # if args.type_learner == 'fgp':
        #     graph_learner = FGP_learner(features, k=args.k, knn_metric=args.sim_function, i=6, sparse=args.sparse)
        # elif args.type_learner == 'mlp':
        #     graph_learner = MLP_learner(nlayers=2, isize=features.shape[1], k=args.k, knn_metric=args.sim_function, i=6, sparse=args.sparse, act=args.activation_learner)
        # elif args.type_learner == 'att':
        #     graph_learner = ATT_learner(nlayers=2, isize=features.shape[1], k=args.k, knn_metric=args.sim_function, i=6, sparse=args.sparse, mlp_act=args.activation_learner)
        # elif args.type_learner == 'gnn':
        #     u_e = torch.from_numpy(np.array(user_edges))
        #     anchor_adj = dgl.graph((u_e[:,0], u_e[:,1]), num_nodes=train_data.user_size, device=args.gpu)
        #     graph_learner = GNN_learner(nlayers=2, isize=features.shape[1], k=args.k, knn_metric=args.sim_function, i=6, sparse=args.sparse, mlp_act=args.activation_learner, adj=anchor_adj)
        # if args.cuda:
        #     features = features.to(args.gpu)
        #     graph_learner = graph_learner.to(args.gpu)
        # optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Use Multi Deepwalk Feat
        multi_deepwalk_feat = None
        if args.use_multi_deepwalk_feat:
            logger.info("use multi_deepwalk_feat")
            multi_deepwalk_feat = []
            for classid in classid2simmat:
                deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, f"{args.dataset}/topic_graph/feature/topicg_deepwalk_topic{classid}.data"), max_idx=train_data.user_size-1)
                multi_deepwalk_feat.append(torch.FloatTensor(deepwalk_feat).unsqueeze(-1))
            deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/deepwalk_emb_user{train_data.user_size}.data"), max_idx=train_data.user_size-1)
            multi_deepwalk_feat.append(torch.FloatTensor(deepwalk_feat).unsqueeze(-1))
            multi_deepwalk_feat = torch.cat(multi_deepwalk_feat, dim=-1)
            if args.cuda:
                multi_deepwalk_feat = multi_deepwalk_feat.to(args.gpu)
        
        new_d = {'hedge_graphs':hedge_graphs, 'multi_deepwalk_feat':multi_deepwalk_feat,}
                #  'user_topic_preference':user_topic_preference, 'diffusion_graph': diffusion_graph, 'features': features, 'graph_learner': graph_learner, 'optimizer_learner': optimizer_learner, }
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)
    
    # elif args.model == 'diffusiongat':
    #     # diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_graph.data"))
    #     diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_motif_graph.data"))
    #     if args.cuda:
    #         diffusion_graph = {key:value.to(args.gpu) for key, value in diffusion_graph.items()}
        
    #     new_d = {'diffusion_graph':diffusion_graph}
    #     train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

    #     model = DiffusionGATNetwork(n_interval=args.n_interval, shape_ret=(64,train_data.user_size), dropout=0.3)

    elif args.model == 'tan':
        opt = Option()
        opt.user_size = train_data.user_size
        opt.device = args.gpu
        new_d = {'opt':opt}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)
        model = TAN(opt)
    
    elif args.model == 'dhgpntm':
        diffusion_graph = LoadDynamicHeteGraph(dataset_dirpath)
        save_pickle(diffusion_graph, os.path.join(dataset_dirpath, "diffusion_graph.data"))
        # diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_graph.data"))
        if args.cuda:
            diffusion_graph = {key:value.to(args.gpu) for key, value in diffusion_graph.items()}
        
        new_d = {'diffusion_graph':diffusion_graph}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

        # model = DyHGCN_H(train_data.user_size, 64, 8)
        model = DyHGCN_H(train_data.user_size, 64, args.n_interval)
    
    elif args.model == 'forest':
        model = RNNModel('GRUCell', train_data.user_size, 64, 64)

    elif args.model == 'ndm':
        model = Decoder(train_data.user_size, d_k=64, d_v=64, d_model=64, d_word_vec=64, d_inner_hid=64, n_head=8, kernel_size=3, dropout=0.1) 
    
    elif args.model == 'hidan':
        # config = Config()
        # config.num_nodes = train_data.user_size
        # model = HiDANModel(config)
        # 初始化 HiDAN 的配置
        hidan_cfg = Config()
        hidan_cfg.num_nodes = train_data.user_size
        # 构造 HiDANModel 时，按照 __init__ 签名传入各个参数
        model = HiDANModel(
            num_nodes      = hidan_cfg.num_nodes,
            embedding_size = hidan_cfg.embedding_size,
            hidden_size    = hidan_cfg.hidden_size,
            n_time_interval= hidan_cfg.n_time_interval,
            keep_prob      = hidan_cfg.dropout,
            l2_weight      = hidan_cfg.l2_weight
        )
    
    elif args.model == 'infvae':
        model = InfVAE(
            user_size=train_data.user_size,
            layer_config=args.infvae_layer_config,
            latent_dim=args.infvae_latent_dim,
            max_seq_len=500,
            dropout=args.dropout,
            neg_samples=args.infvae_neg_samples,
            pos_weight=args.infvae_pos_weight,
            lambda_s=args.infvae_lambda_s,
            lambda_r=args.infvae_lambda_r,
            lambda_p=args.infvae_lambda_p,
        )

    # loss_func = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    loss_func = get_cascade_criterion(train_data.user_size)

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 
        args.d_model, args.n_warmup_steps,)
    patience = EarlyStopping(patience=args.patience)
    
    if args.cuda:
        model = model.to(args.gpu)
        loss_func = loss_func.to(args.gpu)
    
    tensorboard_log_dir = '%s/tensorboard-series/tensorboard_%s_epochs%d' % (os.path.dirname(os.path.abspath(__file__)), args.tensorboard_log, args.epochs)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    shutil.rmtree(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

    t_total = time.time()
    logger.info("training...")
    # epoch_i = 0
    # logger.info("epoch %d, checkpoint!", epoch_i)
    # valid_loss, valid_acc = train(epoch_i, valid_d, graph, model, optimizer, loss_func, writer)
    # logger.info('   - (Validating)    loss: {loss:8.5f}, accuracy: {accu:3.6f} %, gpu memory usage: {mem:3.3f} MiB'.format(
    #     loss=valid_loss, accu=100*valid_acc, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
    
    # start = time.time()
    # scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer)
    # logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
    #     scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
    #     elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
    for epoch_i in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train(epoch_i, train_d, graph, model, optimizer, loss_func, writer, train_data.user_size)
        logger.info('   - (Training)    loss: {loss:8.5f}, accuracy: {accu:3.6f} %, elapse: {elapse:3.3f} min, gpu memory usage: {mem:3.3f} MiB'.format(
            loss=train_loss, accu=100*train_acc, elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
        
        early_stop = patience.step(train_loss.detach().cpu().numpy(), train_acc.detach().cpu().numpy())
        if early_stop:
            start = time.time()
            scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer, train_data.user_size, log_desc='test_')
            logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
                scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            save_model(epoch_i, args, model, optimizer, filepath=os.path.join(DATA_ROOTPATH, f"basic/training/ckpt_model_{args.tensorboard_log}_epoch_{epoch_i}.pkl"))
            break
        
        if (epoch_i + 1) % args.check_point == 0:
            logger.info("epoch %d, checkpoint!", epoch_i)
            # _, valid_loss, valid_acc = evaluate(epoch_i, valid_d, graph, model, optimizer, loss_func, writer, train_data.user_size, log_desc='valid_', return_loss_acc=True)
            valid_loss, valid_acc = train(epoch_i, valid_d, graph, model, optimizer, loss_func, writer, train_data.user_size)
            logger.info('   - (Validating)    loss: {loss:8.5f}, accuracy: {accu:3.6f} %, gpu memory usage: {mem:3.3f} MiB'.format(
                loss=valid_loss, accu=100*valid_acc, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            
            start = time.time()
            scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer, train_data.user_size, log_desc='test_')
            logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
                scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            save_model(epoch_i, args, model, optimizer, filepath=os.path.join(DATA_ROOTPATH, f"basic/training/ckpt_model_{args.tensorboard_log}_epoch_{epoch_i}.pkl"))
                
    logger.info("Total Elapse: {elapse:3.3f} min".format(elapse=(time.time()-t_total)/60))

if __name__ == '__main__':
    main()
