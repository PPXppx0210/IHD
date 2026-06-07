from utils.log import logger
from lib.utils import get_node_types, extend_edges, get_sparse_tensor
import numpy as np
from typing import Dict, List
import os
import random
import datetime
import torch
from torch.utils.data.sampler import Sampler
from scipy.sparse import csr_matrix
import copy
from sklearn.feature_extraction.text import CountVectorizer
from biterm.cbtm import oBTM
from biterm.utility import topic_summuary # helper functions
from itertools import combinations, chain
from numba import njit
from numba import types
from numba.typed import Dict, List
import pyLDAvis.sklearn

class SubGraphSample:
    # NOTE: "Mutable Default Arguments": 
    # 函数的缺省参数是函数定义时作为`__default__`属性附着于函数这个对象上的, 
    # 而如果缺省参数是如list的可变对象, 那么每次调用都会改变它, 也即缺省参数不是每次调用时初始化的, 而是仅在定义时初始化一次
    """
    def foo(a=[]):
        a.append(5)
        return a
    >>> foo()
    [5]
    >>> foo()
    [5, 5]
    >>> foo()
    [5, 5, 5]
    """
    def __init__(self, adj_matrices=None, influence_features=None, vertex_ids=None, labels=None, tags=None, time_stages=None):
        arguments = locals()
        for key, value in arguments.items():
            arguments[key] = [] if value is None else value

        self.adj_matrices = arguments["adj_matrices"]
        self.influence_features = arguments["influence_features"]
        self.vertex_ids = arguments["vertex_ids"]
        self.labels = arguments["labels"]
        self.tags = arguments["tags"]
        self.time_stages = arguments["time_stages"]
    def __len__(self):
        return len(self.labels)

class HeterSubGraphSample:
    def __init__(self, heter_adj_matrices=None, initial_features=None, vertex_ids=None, labels=None):
        arguments = locals()
        for key, value in arguments.items():
            arguments[key] = [] if value is None else value
        
        self.heter_adj_matrices = arguments["heter_adj_matrices"]
        self.initial_features = arguments["initial_features"]
        self.vertex_ids = arguments["vertex_ids"]
        self.labels = arguments["labels"]
    def __len__(self):
        return len(self.labels)

class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0, gap=1):
        self.num_samples = num_samples
        self.start = start
        self.gap = gap

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples, self.gap))

    def __len__(self):
        return self.num_samples // self.gap

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

def gen_user_sets(actionlog, Ntimestages:int, utedges_dirpath: str):
    user_sets = [[set() for _ in range(len(actionlog))] for _ in range(Ntimestages)]

    for hidx, (hashtag, values) in enumerate(actionlog.items()):
        if len(values) == 0:
            continue
        lower_b, upper_b = values[0][1], values[-1][1]
        time_span = (upper_b-lower_b)/(Ntimestages+1)
        elem_idx = 0
        for tidx in range(8):
            start_t, end_t = lower_b+tidx*time_span, lower_b+(tidx+1)*time_span
            while elem_idx < len(values) and values[elem_idx][1] <= end_t:
                user_sets[tidx][hidx].add(values[elem_idx][0])
                elem_idx += 1

    logger.info("User_Sets Num: {}".format(
        " ".join([f"t{tidx}={sum([len(elem) for elem in user_sets[tidx]])}" for tidx in range(Ntimestages)])
    ))
    save_pickle(user_sets, os.path.join(utedges_dirpath, "user_sets.p"))

def add_utedges(data_dirpath:str, utedges_dirpath:str, Ntimestages:int=8):
    # 0. 
    actionlog   = load_pickle(os.path.join(data_dirpath, "ActionLog.p"))
    tweetid2tag = load_pickle(os.path.join(data_dirpath, "text/tweetid2tag.p"))
    node_stat   = load_pickle(os.path.join(data_dirpath, "Node-Stat.p"))
    utedges     = load_pickle(os.path.join(data_dirpath, "graph/U-T.p"))
    tweets      = load_pickle(os.path.join(data_dirpath, "graph/Tweets.p"))
    tweetid2graphid = {value:key for key,value in tweets.items()}
    logger.info("Finish Loading Data...")

    # 1. Get User_Sets Per Time Stage Per Hashtag, i.e. user_sets[tidx][hidx]
    os.makedirs(utedges_dirpath, exist_ok=True)
    if not os.path.exists(os.path.join(utedges_dirpath, "user_sets.p")):
        gen_user_sets(actionlog=actionlog, Ntimestages=Ntimestages, utedges_dirpath=utedges_dirpath)
    user_sets = load_pickle(os.path.join(utedges_dirpath, "user_sets.p"))

    # 2. Gen U-T Edges
    sampled_length = [[] for _ in range(Ntimestages)]
    for tidx in range(Ntimestages):
        ut_edges = []
        for eidx, (user_id, tweet_id) in enumerate(utedges):
            # 1. Get Hashtag which tweet_id belongs to
            # 2. Get Other Users within same hashtag and time-stage
            # 3. Build Additional U-T Edges between current tweet_id and other-users
            if eidx % 10000000 == 0:
                logger.info(f"Enumerating UT-Edges: eidx={eidx}")
            tag = tweetid2tag[int(tweetid2graphid[tweet_id])]
            involved_users = user_sets[tidx][tag]
            if user_id not in involved_users:
                # logger.info(f"Error: user={user_id} not in involved_users={involved_users}")
                continue
            involved_users = [user_id] + random.choices(list(involved_users), k=min(node_stat[user_id]["degree"], node_stat[user_id]["text_num"], len(involved_users)))
            sampled_length[tidx].append(len(involved_users))
            for in_user in involved_users:
                ut_edges.append((in_user, tweet_id))
        logger.info(f"Additional UT-Edges Num={len(ut_edges)}")
        save_pickle(ut_edges, os.path.join(utedges_dirpath, f"ut_edges_t{tidx}.p"))

    # 3. Analyze Distribution of Sampled Involved_Users
    for tidx in range(Ntimestages):
        logger.info(f"Analyzing Distribution of Sampled Involved_Users in Time Stage={tidx}")
        summarize_distribution(sampled_length[tidx])

def build_matrices(graph_dirpath:str, matrices_filepath:str):
    nodes = {
        "User": load_pickle(os.path.join(graph_dirpath, "Users.p")),
        "Tweet": load_pickle(os.path.join(graph_dirpath, "Tweets.p")),
    }
    nodes["ALL"] = nodes["User"] | nodes["Tweet"]
    edges = {
        "U-U": load_pickle(os.path.join(graph_dirpath, "U-U.p")),
        # "U-T": load_pickle(os.path.join(graph_dirpath, "U-T.p")),
        "U-T": load_pickle("/root/Heter-GAT/src/add_utedges_u20000/ut_edges_t0.p")
    }

    start_idx_mp = {}
    indices = 0
    for node_type in ["User", "Tweet"]:
        start_idx_mp[node_type] = indices
        indices += len(nodes[node_type])
    logger.info(f"indices={indices}, start_idx_mp={start_idx_mp}")

    matrices = {"ALL": []}
    for edge_type in ["U-U", "U-T"]:
        node1_t, node2_t = get_node_types(edge_type)
        extended_edges = edges[edge_type]
        if start_idx_mp[node1_t] or start_idx_mp[node2_t]:
            # logger.info(f"node1_t={node1_t}, node2_t={node2_t}, start_idx_node1_t={start_idx_mp[node1_t]}, start_idx_node2_t={start_idx_mp[node2_t]}")
            extended_edges = extend_edges(extended_edges, start_idx_mp[node1_t], start_idx_mp[node2_t])
        matrices[edge_type] = extended_edges
        
        # accumulate all edgelists
        matrices["ALL"] += matrices[edge_type]

    logger.info("Nodes Info: {}".format(" ".join(f"num_{node_t}={len(nodes[node_t])}" for node_t in ["User", "Tweet", "ALL"])))
    logger.info("Edgelists Info: {}".format(" ".join(f"num_{key}={len(value)}" for key, value in matrices.items())))
    os.makedirs(matrices_filepath, exist_ok=True)
    save_pickle(matrices, os.path.join(matrices_filepath, "Edgelist-Matrices.p"))

def set_ar_values(ar, indices, value):
    new_ar = []
    for idx, elem in enumerate(ar):
        new_ar.append(value if idx in indices else elem)
    return new_ar

def find_rt_bound(elem, bound_ar):
    idx = 0
    for rt_bound in bound_ar:
        if elem < rt_bound:
            break
        idx += 1
    return idx if idx<len(bound_ar) else len(bound_ar)-1

def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    uugraph_batch, utgraph_batch, labels_batch, feats_batch = zip(*batch)
    if type(uugraph_batch[0]) == csr_matrix:
        uugraph_batch = torch.stack([get_sparse_tensor(uugraph.tocoo()) for uugraph in uugraph_batch])
    else:
        uugraph_batch = torch.FloatTensor(uugraph_batch)

    if type(utgraph_batch[0]) == csr_matrix:
        utgraph_batch = torch.stack([get_sparse_tensor(utgraph.tocoo()) for utgraph in utgraph_batch])
    else:
        utgraph_batch = torch.FloatTensor(utgraph_batch)
    
    if type(labels_batch[0]).__module__ == 'numpy':
        # NOTE: https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        labels_batch = torch.LongTensor(labels_batch)
    
    if type(feats_batch[0]).__module__ == 'numpy':
        feats_batch = torch.FloatTensor(np.array(feats_batch))
    return uugraph_batch, utgraph_batch, labels_batch, feats_batch

def gen_random_tweet_ids(samples: SubGraphSample, outdir: str, tweets_per_user:int=5):
    tweet_ids = []
    sample_ids = []
    ut_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))

    # TODO: 重构
    for idx in range(len(samples.labels)):
        if idx and idx % 10000 == 0:
            logger.info(f"idx={idx}, sample_ids={len(sample_ids)}, tweet_ids={len(tweet_ids)}")
        stage = samples.time_stages[idx]
        selected_tweet_ids  = set()
        candidate_tweet_ids = set()
        for vertex_id in samples.vertex_ids[idx]:
            available_tweet_ids = ut_mp[stage][vertex_id]
            random_ids = np.random.choice(available_tweet_ids, size=min(tweets_per_user, len(available_tweet_ids)), replace=False)
            selected_tweet_ids  |= set(random_ids)
            candidate_tweet_ids |= set(available_tweet_ids)-set(random_ids)
        candidate_tweet_ids -= selected_tweet_ids
        # logger.info(f"Length: sample={len(selected_tweet_ids)}, remain={len(candidate_tweet_ids)}, expected={len(samples.vertex_ids[idx])*tweets_per_user}")

        if len(selected_tweet_ids) != len(samples.vertex_ids[idx])*tweets_per_user:
            diff = len(samples.vertex_ids[idx])*tweets_per_user - len(selected_tweet_ids)
            if diff > len(candidate_tweet_ids):
                continue
            selected_tweet_ids |= set(np.random.choice(list(candidate_tweet_ids), size=diff, replace=False))
        sample_ids.append(idx)
        tweet_ids.append(selected_tweet_ids)
    logger.info(f"Finish Sampling Random Tweets... sample_ids={len(sample_ids)}, tweet_ids={len(tweet_ids)}")

    os.makedirs(outdir, exist_ok=True)
    selected_samples = SubGraphSample(
        adj_matrices=samples.adj_matrices[sample_ids],
        influence_features=samples.influence_features[sample_ids],
        vertex_ids=samples.vertex_ids[sample_ids],
        labels=samples.labels[sample_ids],
        tags=samples.tags[sample_ids],
        time_stages=samples.time_stages[sample_ids],
    )
    save_pickle(sample_ids, os.path.join(outdir, "sample_ids.p"))
    save_pickle(tweet_ids, os.path.join(outdir, "tweet_ids.p"))
    save_pickle(selected_samples, os.path.join(outdir, "selected_samples.p"))
    logger.info("Finish Saving pkl...")

def extend_subnetwork(file_dir: str):
    hs_filedir = os.path.join(DATA_ROOTPATH, file_dir).replace('stages_', 'hs_')
    samples = load_pickle(os.path.join(hs_filedir, "selected_samples.p"))
    tweet_ids = load_pickle(os.path.join(hs_filedir, "tweet_ids.p"))
    assert len(samples) == len(tweet_ids)

    tweetid2userid_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/tweetid2userid_mp.p"))
    vertex_ids = samples.vertex_ids
    adjs       = samples.adj_matrices
    adjs[adjs != 0] = 1.0
    adjs = adjs.astype(np.dtype('B'))

    extended_vertices, extended_adjs = [], []
    for idx in range(len(samples)):
        subnetwork = np.array(np.concatenate((vertex_ids[idx], np.array(list(tweet_ids[idx])))), dtype=int)
        extended_vertices.append(subnetwork)

        subnetwork_size, num_users = len(subnetwork), len(vertex_ids[idx])
        elem_idx_mp = {elem:idx for idx,elem in enumerate(subnetwork)}
        uu_adj = np.array([[0]*subnetwork_size for _ in range(subnetwork_size)], dtype='B')
        uu_adj[:num_users,:num_users] = adjs[idx]
        # NOTE: Get Corresponding User_id By Tweet_id, and then convert them into indexes in extend_subnetwork
        ut_adj = copy.deepcopy(uu_adj)
        for tweet_id in tweet_ids[idx]:
            user_id = tweetid2userid_mp[tweet_id]
            net_userid = elem_idx_mp[user_id]
            net_tweetid = elem_idx_mp[tweet_id]
            ut_adj[net_userid][net_tweetid] = 1
        extended_adjs.append([uu_adj, ut_adj])
    extended_vertices, extended_adjs = np.array(extended_vertices), np.array(extended_adjs)
    save_pickle(extended_vertices, os.path.join(hs_filedir, "extended_vertices.p"))
    save_pickle(extended_adjs, os.path.join(hs_filedir, "extended_adjs.p"))

def gen_pos_neg_users(g, cascades, sample_ratio, stage):
    pos_users, neg_users = set(), set()
    all_activers = set([elem[0] for elem in cascades])
    max_ts = cascades[0][1] + int((cascades[-1][1]-cascades[0][1]+Ntimestage-1)/Ntimestage) * (stage+1)
    for user, ts in cascades:
        if ts > max_ts:
            break
        # Add Pos Sample
        pos_users.add(user)

        # Choos Neg from Neighborhood
        first_order_neighbor = list(set(g.neighborhood(user, order=1)) - all_activers)
        if len(first_order_neighbor) > 0:
            neg_user = random.choices(first_order_neighbor, k=min(len(first_order_neighbor), sample_ratio))
            neg_users |= set(neg_user)
        else:
            second_order_neighbor = list(set(g.neighborhood(user, order=2)) - all_activers)
            if len(second_order_neighbor) > 0:
                neg_user = random.choices(second_order_neighbor, k=min(len(first_order_neighbor), sample_ratio))
                neg_users |= set(neg_user)
    # logger.info(f"pos={len(pos_users)}, neg={len(neg_users)}, diff={len(pos_users & neg_users)}")
    return pos_users, neg_users

def vec_to_biterms2(A, iA, jA):
    """
    Usage: biterms = vec_to_biterms(dtm.data, dtm.indptr, dtm.indices)
    """
    B_d = []
    for row in range(len(iA)-1):
        b_i = jA[iA[row]:iA[row+1]]
        B_d.append([b for b in combinations(b_i,2)])
    return B_d

@njit
def rand_choice_nb(arr, prob):
    # NOTE: https://github.com/numba/numba/issues/2539
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

class njitBTM:
    """ Biterm Topic Model

        Code and naming is based on this paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
        Thanks to jcapde for providing the code on https://github.com/jcapde/Biterm
    """

    def __init__(self, num_topics, vocab_size, alpha=1., beta=0.01, l=0.5):
        self.K = num_topics
        self.vocab_size = vocab_size
        self.alpha = np.full(self.K, alpha)
        self.beta = np.full((vocab_size, self.K), beta)
        self.l = l
    
    def gibbs(self, iterations):
        return self._gibbs(List(self.B), self.vocab_size, self.K, self.alpha, self.beta, iterations)

    @staticmethod
    @njit
    def _gibbs(B, vocab_size, K, alpha:np.ndarray, beta:np.ndarray, iterations):
        Z = np.zeros(len(B), dtype=np.int8)
        # NOTE: fix error_rewrite(e, 'typing')
        n_wz = np.zeros((vocab_size, K), dtype=np.int32)
        n_z = np.zeros(K, dtype=np.int32)

        for i, b_i in enumerate(B):
            topic = np.random.choice(K, 1)[0]
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic

        for _ in range(iterations):
            for i, b_i in enumerate(B):
                n_wz[b_i[0], Z[i]] -= 1
                n_wz[b_i[1], Z[i]] -= 1
                n_z[Z[i]] -= 1
                P_w0z = (n_wz[b_i[0], :] + beta[b_i[0], :]) / (2 * n_z + beta.sum(axis=0))
                P_w1z = (n_wz[b_i[1], :] + beta[b_i[1], :]) / (2 * n_z + 1 + beta.sum(axis=0))
                P_z = (n_z + alpha) * P_w0z * P_w1z
                P_z = P_z / P_z.sum()
                Z[i] = rand_choice_nb(np.arange(K), P_z)
                n_wz[b_i[0], Z[i]] += 1
                n_wz[b_i[1], Z[i]] += 1
                n_z[Z[i]] += 1
                
        return n_z, n_wz

    def fit(self, B_d, iterations):
        """
        Usage:
        model = njitBTM(num_topics=20, vocab_size=len(vocab))
        step = 1000
        for idx in range(0, len(biterms), step):
            chunk = biterms[idx:idx+step]
            model.fit(chunk, iterations=10)
        """
        self.B = list(chain(*B_d))
        n_z, self.nwz = self.gibbs(iterations)

        self.phi_wz = (self.nwz + self.beta) / np.array([(self.nwz + self.beta).sum(axis=0)] * self.vocab_size)
        self.theta_z = (n_z + self.alpha) / (n_z + self.alpha).sum()

        self.alpha += self.l * n_z
        self.beta += self.l * self.nwz
    
    def transform(self, B_d):
        P_zd = np.zeros([len(B_d), self.K])
        for i, d in enumerate(B_d):
            P_zb = np.zeros([len(d), self.K])
            for j, b in enumerate(d):
                P_zbi = self.theta_z * self.phi_wz[b[0], :] * self.phi_wz[b[1], :]
                P_zb[j] = P_zbi / P_zbi.sum()
            P_zd[i] = P_zb.sum(axis=0) / P_zb.sum(axis=0).sum()

        return P_zd

def btm_model(raw_texts, num_topics, visualize):
    # NOTE: sample a fraction of raw-texts for fitting model, since fitting whole raw-texts(~60m) corpus is toooooo time-consuming!
    docs = sample_docs_foreachuser(docs=raw_texts, user_tweet_mp=load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p")))
    
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(docs)
    biterms = vec_to_biterms2(dtm.data, dtm.indptr, dtm.indices)
    vocab = np.array(cv.get_feature_names_out()) # get_feature_names_out is only available in ver1.0
    model = oBTM(num_topics=num_topics, V=vocab)

    logger.info("Train Online BTM ..")
    step = 1000
    for idx in range(0, len(biterms), step):
        chunk = biterms[idx:idx+step]
        model.fit(chunk, iterations=10)
    
    topic_distr = model.transform(biterms)
    logger.info(f"Finish Training...Calculating Topic Coherence")
    topic_summuary(P_wz=model.phi_wz.T, X=dtm, V=vocab, M=10)
    if visualize:
        dtm_dense = dtm.toarray()
        panel = pyLDAvis.prepare(model.phi_wz.T, topic_distr, np.count_nonzero(dtm_dense, axis=1), vocab, np.sum(dtm_dense, axis=0))
    else:
        panel = None
    return {
        "topic-distr": topic_distr,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": panel,
    }

@njit
def score2(doc, m_z, n_z, n_z_w, V, D, alpha=0.01, beta=0.01, K=20):
    '''
    Score a document

    Implements formula (3) of Yin and Wang 2014.
    http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

    :param doc: list[str]: The doc token stream
    :return: list[float]: A length K probability vector where each component represents
                            the probability of the document appearing in a particular cluster
    '''
    p = [0 for _ in range(K)]

    #  We break the formula into the following pieces
    #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
    #  lN1 = log(m_z[z] + alpha)
    #  lN2 = log(D - 1 + K*alpha)
    #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
    #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

    lD1 = np.log(D - 1 + K * alpha)
    doc_size = len(doc)
    for label in range(K):
        lN1 = np.log(m_z[label] + alpha)
        lN2 = 0
        lD2 = 0
        for word in doc:
            lN2 += np.log(n_z_w[label].get(word, 0) + beta)
        for j in range(1, doc_size +1):
            lD2 += np.log(n_z[label] + V * beta + j - 1)
        p[label] = np.exp(lN1 - lD1 + lN2 - lD2)

    # normalize the probability vector
    pnorm = sum(p)
    pnorm = pnorm if pnorm>0 else 1
    return [pp/pnorm for pp in p]

def gsdmm_score_topic_distr(docs, m_z, n_z, n_z_w, n_terms, n_docs):
    # Preparation
    n_z_w_param = []
    for n_z_w_element in n_z_w:
        dict_param_t = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        keys = list(n_z_w_element.keys())
        for key in keys:
            dict_param_t[key] = n_z_w_element[key]
        n_z_w_param.append(dict_param_t)
    
    scores = []
    # NOTE: time-consuming!!! ~1min/100,000docs in the following for-loop
    for doc in docs:
        scores.append(score2(doc, List(m_z), List(n_z), List(n_z_w_param), n_terms, n_docs))
    return scores
