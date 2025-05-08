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
# from src.sota.TAN.model import TAN
# from src.sota.TAN.Option import Option
# from src.sota.DHGPNTM.DyHGCN import DyHGCN_H
# from src.sota.DHGPNTM.DataConstruct import LoadDynamicHeteGraph
# from src.sota.FOREST.model import RNNModel
# from src.sota.NDM.transformer.Models import Decoder
# from src.sota.HiDAN.model import HiDANModel
# from src.sota.HiDAN.config import Config
# from src.sota.MSHGAT.model import MSHGAT
# from src.sota.MSHGAT.graphConstruct import ConRelationGraph, ConHyperGraphList
# from src.sota.MSHGAT.dataLoader import Split_data
# from src.sota.MINDS.Module import MINDS
# from src.sota.MINDS.HypergraphUtil import DynamicCasHypergraph
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
# import dhg

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_printoptions(threshold=10_000)

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--dataset', type=str, default='Twitter-Huangxin', help="available options are ['Weibo-Aminer','Twitter-Huangxin']")
parser.add_argument('--model', type=str, default='heteredgegat', help="available options are ['densegat','heteredgegat','diffusiongat','dhgpntm']")
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
parser.add_argument('--sparsity', type=int, default=100, help='Sparsity Comparison (0-100)')

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
    writer.add_scalar(log_desc+'acc',  n_correct/total, epoch_i+1)
    torch.cuda.empty_cache() # 清理缓存，让 cuda reserved 数字回落
    return loss/total, n_correct/total

def evaluate(epoch_i, data, graph, model, optimizer, loss_func, writer, user_size, k_list=[10,50,100], log_desc='valid_'):
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
    
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1); writer.add_scalar(log_desc+'acc', correct/total, epoch_i+1); writer.add_scalar(log_desc+'mrr', scores['MRR'], epoch_i+1)
    writer.add_scalar(log_desc+'hits@10',  scores['hits@10'],  epoch_i+1);  writer.add_scalar(log_desc+'map@10',  scores['map@10'],  epoch_i+1)
    writer.add_scalar(log_desc+'hits@50',  scores['hits@50'],  epoch_i+1);  writer.add_scalar(log_desc+'map@50',  scores['map@50'],  epoch_i+1)
    writer.add_scalar(log_desc+'hits@100', scores['hits@100'], epoch_i+1);  writer.add_scalar(log_desc+'map@100', scores['map@100'], epoch_i+1)
    # writer.add_scalar(log_desc+'auc', auc, epoch_i+1);                    writer.add_scalar(log_desc+'f1', f1, epoch_i+1)
    # writer.add_scalar(log_desc+'prec', prec, epoch_i+1);                  writer.add_scalar(log_desc+'rec', rec, epoch_i+1)
    torch.cuda.empty_cache()
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

    dataset_dirpath = f"{DATA_ROOTPATH}/{args.dataset}"
    # if args.dataset == 'Twitter-Huangxin':
    #     dataset_dirpath += '/sub10000'
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
        # u2idx_filepath, idx2u_filepath = f"{dataset_dirpath}/u2idx.data", f"{dataset_dirpath}/idx2u.data"
        # _u2idx = load_pickle(u2idx_filepath)
        # mapped_edges = [
        #     (_u2idx[u], _u2idx[v])
        #     for u, v in user_edges
        #     if u in _u2idx and v in _u2idx
        # ]
        # user_edges = mapped_edges
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

    if args.model == 'densegat':
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

        save_pickle(hypergraph_list, os.path.join(dataset_dirpath, "/hypergraph_list.data"))
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
        base_filename = "new_topic_diffusion_graph"
        if args.use_random_multiedge:
            base_filename += "_random"
        elif args.use_motif:
            base_filename = "topic_diffusion_motif_graph"
        else:
            # base_filename += "_full"
            base_filename += "_{}".format(args.graph_filename)
        
        sparsity_suffix = ""
        if args.sparsity < 100:
            sparsity_suffix = f"_sparsity{args.sparsity}.0"
        
        classid2simmat = load_pickle(os.path.join(dataset_dirpath, f"output/{base_filename}_windowsize{args.window_size}{sparsity_suffix}.data"))
        # if args.cuda:
        #     classid2simmat = {classid:simmat.to(args.gpu) for classid, simmat in classid2simmat.items()}
        
        # n_adj = max(classid2simmat.keys())+1
        
        topk = args.graph_topk
        n_adj = min(len(classid2simmat), topk)
        logger.info("n_adj : {}".format(n_adj))
        if args.tweet2graph == 0:
            # hedge_graphs = [graph] * (n_adj+1)
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
        save_pickle(diffusion_graph, os.path.join(dataset_dirpath, "/diffusion_graph.data"))
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
        config = Config()
        config.num_nodes = train_data.user_size
        model = HiDANModel(config)

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
            scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer, train_data.user_size)
            logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
                scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            save_model(epoch_i, args, model, optimizer, filepath=os.path.join(DATA_ROOTPATH, f"basic/training/ckpt_model_{args.tensorboard_log}_epoch_{epoch_i}.pkl"))
            break
        
        if (epoch_i + 1) % args.check_point == 0:
            logger.info("epoch %d, checkpoint!", epoch_i)
            valid_loss, valid_acc = train(epoch_i, valid_d, graph, model, optimizer, loss_func, writer, train_data.user_size)
            logger.info('   - (Validating)    loss: {loss:8.5f}, accuracy: {accu:3.6f} %, gpu memory usage: {mem:3.3f} MiB'.format(
                loss=valid_loss, accu=100*valid_acc, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            
            start = time.time()
            scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer, train_data.user_size)
            logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
                scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            save_model(epoch_i, args, model, optimizer, filepath=os.path.join(DATA_ROOTPATH, f"basic/training/ckpt_model_{args.tensorboard_log}_epoch_{epoch_i}.pkl"))
                
    logger.info("Total Elapse: {elapse:3.3f} min".format(elapse=(time.time()-t_total)/60))

if __name__ == '__main__':
    main()
