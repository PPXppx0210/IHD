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


def get_scores(pred_cascade:torch.Tensor, gold_cascade:torch.Tensor, k_list=[10,50,100]):
    gold_cascade = gold_cascade.contiguous().view(-1)           # (#samples,)
    pred_cascade = pred_cascade.view(gold_cascade.size(0), -1)  # (#samples, #user_size)
    pred_cascade = pred_cascade.detach().cpu().numpy()
    gold_cascade = gold_cascade.detach().cpu().numpy()
    scores = compute_metrics(pred_cascade, gold_cascade, k_list)
    return scores


def train(epoch_i, data, graph, model, optimizer, loss_func, writer, user_size, log_desc='train_'):
    model.train()
    
    loss, correct, total = 0., 0., 0.
    for _, batch in enumerate(data['batch']):

        cas_users, cas_tss, cas_intervals, cas_classids, cas_idx, tgt_len = batch
        if args.cuda:
            cas_users = cas_users.to(args.gpu)
            cas_tss = cas_tss.to(args.gpu)
            cas_intervals = cas_intervals.to(args.gpu)
            cas_idx = cas_idx.to(args.gpu)
            tgt_len = tgt_len.to(args.gpu)
        gold_cascade = cas_users[:, 1:]
        
        optimizer.zero_grad()
        if args.model == 'lhd':
            pred_cascade = model(cas_users, cas_intervals, cas_classids, data['hedge_graphs'], feats=data['feat'], multi_deepwalk_feat=data['multi_deepwalk_feat'])
        
        loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
        loss_batch.backward()
        optimizer.step()
        optimizer.update_learning_rate()

        n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
        loss += loss_batch.item()
        # logger.info(f"loss={loss}")
        correct += n_correct.item()
        total += n_words

        del pred_cascade
    
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1)
    writer.add_scalar(log_desc+'acc',  n_correct/total, epoch_i+1)
    torch.cuda.empty_cache() 
    return loss/total, n_correct/total

def evaluate(epoch_i, data, graph, model, optimizer, loss_func, writer, user_size, k_list=[10,50,100], log_desc='valid_'):
    model.eval()
    
    loss, correct, total = 0., 0., 0.
    scores = {'MRR': 0,}
    for k in k_list:
        scores[f'hits@{k}'] = 0
        scores[f'map@{k}'] = 0
    
    with torch.no_grad():
        for _, batch in enumerate((data['batch'])):

            cas_users, cas_tss, cas_intervals, cas_classids, cas_idx, tgt_len = batch
            if args.cuda:
                cas_users = cas_users.to(args.gpu)
                cas_tss = cas_tss.to(args.gpu)
                cas_intervals = cas_intervals.to(args.gpu)
                cas_idx = cas_idx.to(args.gpu)
                tgt_len = tgt_len.to(args.gpu)
            gold_cascade = cas_users[:, 1:]
            
            optimizer.zero_grad()
            if args.model == 'lhd':
                pred_cascade = model(cas_users, cas_intervals, cas_classids, data['hedge_graphs'], feats=data['feat'], multi_deepwalk_feat=data['multi_deepwalk_feat'])
            
            loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
                    
            n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
            loss += loss_batch.item()
            correct += n_correct.item()
            total += n_words

            scores_batch = get_scores(pred_cascade, gold_cascade, k_list)
            
            scores['MRR'] += scores_batch['MRR'] * n_words
            for k in k_list:
                scores[f'hits@{k}'] += scores_batch[f'hits@{k}'] * n_words
                scores[f'map@{k}'] += scores_batch[f'map@{k}'] * n_words
    
    model.train()
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


def main():
    # torch.set_num_threads(4)

    dataset_dirpath = f"{DATA_ROOTPATH}/{args.dataset}"
    
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

    user_edges = load_pickle(os.path.join(dataset_dirpath, "edges.data"))
    user_edges = user_edges + [(i,i) for i in range(train_data.user_size)]
    user_edges = list(zip(*user_edges))
    edges_t = torch.LongTensor(user_edges) # (2,#num_edges)
    weight_t = torch.FloatTensor([1]*edges_t.size(1))
    graph = Data(edge_index=edges_t, edge_weight=weight_t)
    
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
        
            if args.cuda:
                tweet_side_emb = tweet_side_emb.to(args.gpu)
            
            user_side_emb = torch.cat([user_side_emb, tweet_side_emb], dim=1)
        
        new_d = {'feat': user_side_emb}
    train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

    
    if args.model == 'lhd':
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
        model = LHD(user_size=train_data.user_size, n_feat=n_feat, n_adj=n_adj, num_interval=args.n_interval, n_comp=args.n_component, 
            n_units=n_units, n_heads=n_heads, attn_dropout=args.attn_dropout, dropout=args.dropout, 
            use_gat=args.use_gat, use_time_decay=args.use_time_decay, 
            use_add_attn=use_add_attn, use_topic_selection=use_topic_selection, random_feat_dim=random_feat_dim)

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
