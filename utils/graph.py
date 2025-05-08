from utils.utils import load_pickle, save_pickle, flattern, DATA_ROOTPATH
from utils.log import logger
import igraph
import os
import random
import re
import torch
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Any, Dict, List, Tuple, Union
from itertools import combinations
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

def init_graph(nb_nodes:int, edgelist:List[Any], is_directed:bool=False, outputfile_dirpath:str="", save_graph:bool=False):
    graph = igraph.Graph(nb_nodes, directed=is_directed)
    graph.add_edges(edgelist)
    if not is_directed:
        graph.to_undirected()
    graph.simplify()

    if save_graph:
        # Save Graph in IGraph Format
        os.makedirs(outputfile_dirpath, exist_ok=True)
        # with open(os.path.join(outputfile_dirpath, "igraph_edgelist"), "w") as f:
        #     graph.write(f, format="edgelist")
        save_pickle(graph, "{}/igraph-{}.p".format(outputfile_dirpath, "directed" if is_directed else "undirected"))
        logger.info("Save Network in IGraph Format to {}/igraph-{}.p".format(outputfile_dirpath, "directed" if is_directed else "undirected"))

    return graph

def gen_subgraph(graph_dirpath:str, nb_users:int, subgraph_dirpath:str):
    nodes = {
        "User": load_pickle(os.path.join(graph_dirpath, "Users.p")),
        "Tweet": load_pickle(os.path.join(graph_dirpath, "Tweets.p")),
    }
    edges = {
        "U-U": load_pickle(os.path.join(graph_dirpath, "U-U.p")),
        "U-T": load_pickle(os.path.join(graph_dirpath, "U-T.p")),
    }

    sample_nodes = {"User": {}, "Tweet": {}}
    sample_edges = {"U-U":  [], "U-T":   []}
    os.makedirs(subgraph_dirpath, exist_ok=True)
    
    # Sample User Nodes
    for user_id, graphid in nodes["User"].items():
        if graphid < nb_users:
            sample_nodes["User"][user_id] = graphid
    # save_pickle(sample_nodes["User"], os.path.join(subgraph_dirpath, f"Users_u{nb_users}.p"))
    save_pickle(sample_nodes["User"], os.path.join(subgraph_dirpath, f"Users.p"))
    
    # Sample User-User Edges
    for user1, user2 in edges["U-U"]:
        if user1 < nb_users and user2 < nb_users:
            sample_edges["U-U"].append((user1, user2))
    save_pickle(sample_edges["U-U"], os.path.join(subgraph_dirpath, f"U-U.p"))

    # Sample User-Tweet Edges
    for user_id, tweet_id in edges["U-T"]:
        if user_id < nb_users:
            sample_edges["U-T"].append((user_id, tweet_id))
    save_pickle(sample_edges["U-T"], os.path.join(subgraph_dirpath, f"U-T.p"))

    # Sample Tweet Nodes
    nb_tweets = len(sample_edges["U-T"])
    for tweet_id, graphid in nodes["Tweet"].items():
        if graphid < nb_tweets:
            sample_nodes["Tweet"][tweet_id] = graphid
    save_pickle(sample_nodes["Tweet"], os.path.join(subgraph_dirpath, f"Tweets.p"))

    logger.info(f"Nodes Info: Users={nb_users}, Tweets={nb_tweets}, Total={nb_users+nb_tweets}")
    logger.info("Edges Info: U-U={}, U-T={}, Total={}".format(len(sample_edges["U-U"]), len(sample_edges["U-T"]), len(sample_edges["U-U"])+len(sample_edges["U-T"])))

def gen_subactionlog(actionlog_dirpath:str, nb_users:int, subactionlog_dirpath:str):
    actionlog = load_pickle(os.path.join(actionlog_dirpath, "ActionLog.p"))
    subactionlog = {}

    for hashtag, values in actionlog.items():
        subvalues = []
        for graphid, timestamp in values:
            if graphid < nb_users:
                subvalues.append((graphid, timestamp))
        subactionlog[hashtag] = subvalues
    save_pickle(subactionlog, os.path.join(subactionlog_dirpath, f"ActionLog_u{nb_users}.p"))

def sample_tweets_around_user(users:set, ut_mp:dict, tweets_per_user:int, counts_matter:bool=False, return_edges:bool=False):
    """
    功能: 根据ut_mp为users中的每个user挑选min(tweets_per_user, len(ut_mp[user]))个推文邻居节点
    参数: 
        counts_matter表示总共挑选的推文节点数量必须等于len(users)*tweets_per_user,不满足则返回空推文节点集合
    返回:
        tweet_nodes(set), enough_tweet_nodes(bool,counts_matter=True), ut_edges(list,return_edges=True)
    """
    tweets, remaining_tweets, ut_edges = set(), set(), set()
    for user in users:
        selected_tweets = random.choices(ut_mp[user], k=min(len(ut_mp[user]), tweets_per_user))
        tweets |= set(selected_tweets)
        if counts_matter:
            remaining_tweets |= set(ut_mp[user]) - set(selected_tweets)
        if return_edges:
            for tweet in selected_tweets:
                ut_edges.add((user, tweet))
    
    return tweets, remaining_tweets, list(ut_edges)

def reindex_graph(old_nodes:list, old_edges:list, add_self_loop:bool=True):
    nodes = {}
    node_indices, max_indices = 0, 0
    for nodes_l in old_nodes:
        for node in nodes_l:
            nodes[node+max_indices] = node_indices
            node_indices += 1
        max_indices = max(list(nodes.keys()))+1
    
    edges = [[], []]
    for from_, to_ in old_edges[0]:
        edges[0].append((from_, to_))
    offset = max(old_nodes[0])+1
    for from_, to_ in old_edges[1]:
        edges[1].append((nodes[from_], nodes[to_+offset]))
    
    if add_self_loop:
        for node in range(len(old_nodes[0])):
            edges[0].append((node, node))
        for node in range(len(old_nodes[0]), node_indices):
            edges[1].append((node, node))

    return nodes, edges

def build_meta_relation(relations, nb_users):
    meta_relations = []
    for relation in relations:
        meta_relations.append(np.matmul(relation[:nb_users], relation[:nb_users].T))
    return meta_relations

def create_sparsemat_from_edgelist(edgelist, m, n):
    if not isinstance(edgelist, np.ndarray):
        edgelist = np.array(edgelist)
    rows, cols = edgelist[:,0], edgelist[:,1]
    ones = np.ones(len(rows), np.uint8)
    mat = sparse.coo_matrix((ones, (rows, cols)), shape=(m, n))
    return mat.tocsr()

def create_adjmat_from_edgelist(edgelist, size):
    adjmat = [[0]*size for _ in range(size)]
    for from_, to_ in edgelist:
        adjmat[from_][to_] = 1
    return np.array(adjmat, dtype=np.uint8)

def extend_featspace(feats: List[np.ndarray]):
    """
    Func: [Nu*fu,Nt*ft] -> (Nu+Nt)*(fu+ft)
    Solu: concat each feat-space, fill other positions with zero
    """
    full_dim  = sum([feat.shape[1] for feat in feats], 0)
    front_dim = 0
    extend_feats = []
    for feat in feats:
        nb_node = feat.shape[0]
        extend_feat = np.concatenate(
            (np.zeros(shape=(nb_node,front_dim)), feat, np.zeros(shape=(nb_node,full_dim-front_dim-feat.shape[1])))
        , axis=1)
        extend_feats.append(extend_feat)
        front_dim += feat.shape[1]
    return np.vstack(extend_feats)

def extend_featspace2(feats: List[np.ndarray]):
    """
    Func: [Nu*fu,Nt*ft] -> [(Nu+Nt)*fu,(Nu+Nt)*ft]
    """
    nb_nodes = sum([feat.shape[0] for feat in feats], 0)
    extend_feats = []
    front_dim = 0
    for feat in feats:
        extend_feat = np.concatenate(
            (np.zeros(shape=(front_dim, feat.shape[1])), feat, np.zeros(shape=(nb_nodes-front_dim-feat.shape[0], feat.shape[1])))
            , axis=0)
        extend_feats.append(extend_feat)
        front_dim += feat.shape[0]
    return extend_feats

def extend_wholegraph(g, ut_mp, initial_feats, tweet_per_user=20, sparse_graph=True):
    """
    功能: 在subg_deg483子图和ut_mp的基础上, 根据tweet_per_user参数重新生成hadjs和feats
    参数: sparse_graph=False时, 返回的实际上是一同质图
    """
    user_nodes = g.vs["label"]
    tweet_nodes, _, ut_edges = sample_tweets_around_user(users=set(user_nodes), ut_mp=ut_mp, tweets_per_user=tweet_per_user, return_edges=True)
    tweet_nodes = list(tweet_nodes)
    logger.info(f"nb_users={len(user_nodes)}, nb_tweets={len(tweet_nodes)}")

    # Users: 44896, Tweets: 10008103, Total: 10052999
    nodes, edges = reindex_graph([user_nodes, tweet_nodes], [g.get_edgelist(), ut_edges])

    if sparse_graph:
        uu_mat = create_sparsemat_from_edgelist(edges[0], len(nodes), len(nodes))
        ut_mat = create_sparsemat_from_edgelist(edges[1], len(nodes), len(nodes))
        hadjs = [uu_mat, ut_mat]
    else:
        hadjs = build_meta_relation([create_adjmat_from_edgelist(edges[0], len(nodes)), create_adjmat_from_edgelist(edges[1], len(nodes))], nb_users=len(user_nodes))

    user_feats, tweet_feats = initial_feats
    if len(user_feats) != len(user_nodes):
        user_feats = user_feats[user_nodes]
    tweet_feats = tweet_feats[tweet_nodes]

    if sparse_graph:
        feats = np.concatenate((
            np.append(user_feats, np.zeros(shape=(user_feats.shape[0], tweet_feats.shape[1])),  axis=1), 
            np.append(np.zeros(shape=(tweet_feats.shape[0], user_feats.shape[1])), tweet_feats, axis=1), 
        ), axis=0)
    else:
        feats = user_feats
    return hadjs, feats

def unique_cascades(df):
    unique_df = {}
    for hashtag, cascades in df.items():
        unique_cs, us = [], set()
        for user, timestamp in cascades:
            if user in us:
                continue
            us.add(user)
            unique_cs.append((user,timestamp))
        unique_df[hashtag] = unique_cs
    return unique_df

def gen_pos_neg_users2(g, cascades, sample_ratio):
    pos_users, neg_users = set(), set()
    all_activers = set([elem[0] for elem in cascades])
    for user, _ in cascades:
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

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def preprocess_timelines_byusernum(timelines:Dict[int,list], min_user_participate:int, max_user_participate:int)->Dict[int,list]:
    # Sort by Timestamp
    timelines = {key:sorted(value, key=lambda elem:elem[1]) for key,value in timelines.items()}
    # Remove Too Short or Too Long Cascades By Unique User-Nums
    user_participate_mp = {key:len(set([elem[0] for elem in value])) for key,value in timelines.items()}
    if max_user_participate < min_user_participate:
        max_user_participate = max([elem for elem in user_participate_mp.values()])
    timelines = {key:value for key,value in timelines.items() if user_participate_mp[key]>=min_user_participate and user_participate_mp[key]<=max_user_participate}
    return timelines

def search_cascades(user_texts_mp:Dict[int,dict], subg:igraph.Graph)->Dict[int,list]:
    """
    功能: 将符合相同条件的推文组织为同一话题级联, 这些条件包括,
        1. 包含相同tag, i.e. #blacklivesmatter
        2. 包含相同url, i.e. 
        3. 用"RT @USER"形式标识的多个文本内容相同的推文
    """
    userid2graphid = {user_id:graph_id for graph_id, user_id in enumerate(subg.vs["label"])}
    timelines_tag = aggby_same_text(user_texts_mp, regex_pattern=r'(#\w+)', userid2graphid=userid2graphid)
    timelines_url = aggby_same_text(user_texts_mp, regex_pattern=r'(https?://[a-zA-Z0-9.?/&=:]+)', userid2graphid=userid2graphid)

    timelines_tag = preprocess_timelines_byusernum(timelines_tag, min_user_participate=5, max_user_participate=2000)
    timelines_url = preprocess_timelines_byusernum(timelines_url, min_user_participate=5, max_user_participate=-1)
    # NOTE: 'RT @USER: '这一方式得到的级联基本被TAG级联和URL级联所覆盖, 因此不再采用这种方式构造级联
    # timelines_retweet = aggby_retweet_info(user_texts_mp, userid2graphid)
    return timelines_tag, timelines_url

def aggby_same_text(user_texts_mp:Dict[int,dict], regex_pattern:str, userid2graphid:Dict[int,int])->Dict[int,list]:
    """
    参数: 
        user_texts_mp: { user_id: { text_id: {'timestamp': , 'text': ,}, { text_id2: {} }, ... }, ... }
        regex_pattern: 过滤话题级联的正则表达式模式
            i.e. tag_pattern = r'(#\w+)', url_pattern = r'(https?://[a-zA-Z0-9.?/&=:]+)'
        userid2graphid: 原始拓扑图(v=208894)与采样子图(v=44896)的节点ID对应关系
            i.e. userid2graphid={user_id:graph_id for graph_id, user_id in enumerate(subg.vs["label"])}
    返回值:
        timelines: { tag: [(u1,t1),...], tag2:... }
    """
    valid_users = list(userid2graphid.keys())
    timelines:Dict[int,list] = {}

    for user_id, raw_texts in user_texts_mp.items():
        if user_id not in valid_users:
            continue
        for _, text in raw_texts.items():
            group = re.finditer(regex_pattern, text['text'])
            if group is None:
                continue
            for match in group:
                end_pos = match.span()[1]
                # incomplete tags shouldnt been recorded
                if end_pos == len(text['text'])-1 and text['text'][-1] == '…':
                    continue
                tag = match.group(1).lower()
                tag = tag.replace('…', '') # remove not filtered '…'
                if tag not in timelines:
                    timelines[tag] = []
                timelines[tag].append((userid2graphid[user_id], text['timestamp']))
    return timelines

def aggby_retweet_info(user_texts_mp:Dict[int,dict], userid2graphid:Dict[int,int]):
    """
    思路: 将去掉"RT @USER: "形式后, 包含相同文本内容的推文组织成级联
        1. 先将包含"RT @USER: "形式的推文找出来, 并聚合形成级联;
        2. 再针对每个上述级联, 暴力搜索所有推文, 查找是否存在这些转发推文的原始推文, 如果存在则加入该级联中.
    """
    # Aggregate RT Info
    retweet_pattern = r'RT @\w+: (.+)'
    timelines = aggby_same_text(user_texts_mp=user_texts_mp, regex_pattern=retweet_pattern, userid2graphid=userid2graphid)
    timelines = {key:value for key,value in timelines.items() if key[0]!='#' and len(key)>15}
    timelines = preprocess_timelines_byusernum(timelines, min_user_participate=100, max_user_participate=0)

    # Search for Possible Original Tweets
    valid_users = list(userid2graphid.keys())
    for retweet, _ in timelines.items():
        for user_id, raw_texts in user_texts_mp.items():
            if user_id not in valid_users:
                continue
            for _, text in raw_texts.items():
                if text['text'].lower() == retweet:
                    timelines[retweet].append((userid2graphid[user_id], text['timestamp']))
                    logger.info(f"retweet={retweet}, length={len(timelines[retweet])}")

    return timelines

def find_tweet_by_cascade_info(user_texts_mp:Dict[int,dict], user_id:int, timestamp:int)->str:
    """
    Usage: find_tweet_by_cascade_info(user_texts_mp=user_texts, user_id=subg.vs[user]["label"], timestamp=ts)
    """
    texts = user_texts_mp[user_id]
    for _, text in texts.items():
        if text['timestamp'] == timestamp:
            return text['text']
    return None

def aggregate_texts_by_timeline(cascades:Dict[int,list], user_texts_mp:Dict[int,dict], g:igraph.Graph)->Dict[int,list]:
    # Find Tweet, and Build (tid) -> (text), (tid) -> (uid, timestamp)
    tid = 0
    tid2text_mp = {}
    tid2caselem_mp = {}
    tids_aggby_timeline_mp = {}

    for tag, cascades in tqdm(cascades.items()):
        tids = []
        for uid, timestamp in cascades:
            text = find_tweet_by_cascade_info(user_texts_mp, user_id=g.vs["label"][uid], timestamp=timestamp)
            if text is None:
                continue
            tid2text_mp[tid] = text
            tid2caselem_mp[tid] = {"uid":uid, "ts":timestamp}
            tids.append(tid)
            tid += 1
        tids_aggby_timeline_mp[tag] = tids

    # save_pickle(tids_aggby_timeline_mp, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/tids_aggby_timeline_mp.pkl")
    # save_pickle(tid2text_mp, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/tid2text_mp.pkl")
    # save_pickle(tid2caselem_mp, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/tid2caselem_mp.pkl")
    logger.info("Completed...")

    return tids_aggby_timeline_mp

def build_tweet_mat_for_cascades(tweet_mat_shape:Tuple[int,int], cascades:Dict[str,list], tweet_features:np.ndarray)->Dict[str,np.ndarray]:
    tag2tweet_mat_mp = {}
    tid = 0
    for tag, cascade in cascades.items():
        # Build One-to-One Relationship, UID <-> TID
        # Keep the First User Appearance when coming to Multiple Appearances in One Cascade
        uids, tids, uset = [], [], set()
        for idx, (uid, _) in enumerate(cascade):
            if uid in uset: continue
            uids.append(uid); tids.append(tid+idx); uset.add(uid)
        tid += len(cascade)

        # Build Tweet Mat
        tweet_mat = np.zeros(shape=tweet_mat_shape)
        tweet_mat[uids, :] = tweet_features[tids, :]
        tag2tweet_mat_mp[tag] = tweet_mat
    return tag2tweet_mat_mp

def mask_mat(mat:Union[torch.Tensor,np.ndarray], mask:torch.Tensor)->torch.Tensor:
    if type(mat) == np.ndarray:
        mat = torch.tensor(mat)
    if len(mat.shape) > len(mask.shape) or mat.shape[1] > mask.shape[1]:
        for _ in range(mask.dim(), mat.dim()):
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(mat)
    masked_mat = mat.data.masked_fill(~mask.bool(), 0.0)
    return masked_mat.float()

def build_topic_similarity_user_edges(ut_mp:dict):
    # 1. Get Doc-Topic
    suffix = f"remove_hyphen_reduce_auto"
    docs = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_process_remove_hyphen.pkl")
    docs = flattern(docs)
    topic_labels = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label_{suffix}.pkl")
    topic_probs  = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob_{suffix}.pkl")
    topic_reprs  = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_representation_{suffix}.pkl")

    # ut_mp = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/ut_mp_filter_lt2words_processedforbert_subg.pkl")
    user_ids = [[user]*len(tweet_ids) for user, tweet_ids in ut_mp.items()]
    user_ids = flattern(user_ids)
    n_user = len(ut_mp)

    # 2. 
    df = pd.DataFrame({
        'Doc'       : docs,
        'Topic_ID'  : topic_labels,
        'Topic_Prob': topic_probs,
        'User_ID'   : user_ids,
    })
    df['Top_N_Words'] = df['Topic'].map(topic_reprs)

    # 3. Filter
    # 3.1 Remove those topics spanning across too many users
    # Aggregate Users by Topic-Labels
    agg_user_dict = df.groupby('Topic_ID').agg({'User_ID': set})['User_ID'].to_dict()
    # df['Agg_Topic_ID'] = df['User_ID'].map(agg_topic_dict)
    too_many_user_topics = [topic for topic,users in agg_user_dict.items() if len(users)>int(0.5*n_user)]
    logger.info(f"Remove Topics spanning across too many users... Topics={too_many_user_topics}")
    df = df[~df['Topic_ID'].isin(too_many_user_topics)]
    # df_1 = df[~df['Topic_ID'].isin(too_many_user_topics)]
    # logger.info(f"Docs Number Before={len(df)}, After={len(df_1)}")

    # 3.2 Remove those topics including too few tweets
    tweet_num_per_topic = df.groupby(['Topic_ID'])['Topic_ID'].count().to_dict()
    too_few_tweet_topics = [topic for topic,tweet_num in tweet_num_per_topic.items() if tweet_num<10]
    logger.info(f"Remove Topics including too few tweets... Topics={too_few_tweet_topics}")
    df = df[~df['Topic_ID'].isin(too_few_tweet_topics)]

    # 3.3 Remove those users not having strong association with other users within topics
    # 3.4 Remove those short-period topics

    # 4. Build Full-Connected User-User Edges
    user_edges_dict = {}
    agg_user_dict = df.groupby('Topic_ID').agg({'User_ID': set})['User_ID'].to_dict()
    for key, value in agg_user_dict.items():
        user_edges_dict[key] = list(combinations(value, r=2))
    
    return user_edges_dict

def build_sim_edges(data_dict:dict, window_size:int)->dict:
    n_cluster = max([max(elem['label']) for elem in data_dict.values()])+1
    tag2simedges_mp = {}

    for tag, cascades in data_dict.items():
        # Only Keep First Appearance Foreach User
        uni_users, uni_labels = [], []
        us = set()
        for user, label in zip(cascades['user'], cascades['label']):
            if user in us: continue
            us.add(user)
            uni_users.append(user); uni_labels.append(label)
        assert len(uni_users) == len(uni_labels)

        # Aggregate SimEdges Foreach Class
        simedges = {key:[] for key in range(n_cluster)}
        
        for i in range(len(uni_labels)-1):
            for j in range(max(0,i-1-window_size),min(i+1+window_size,len(uni_labels))): # (i-ws-1<-i->i+ws+1)
                if uni_labels[i] == uni_labels[j]:
                    simedges[uni_labels[i]].append((uni_users[i],uni_users[j]))
        
        tag2simedges_mp[tag] = simedges
    
    return tag2simedges_mp

def find_prominent_components_foreach_tag(tag2simedges_mp:dict, n_component:int=3, max_ratio:float=0.8)->dict:
    # Find Most Prominent Component Topic Foreach Timeline
    tagid2classids = {}
    for tag, simedges in tag2simedges_mp.items():
        ratios = [len(elem) for elem in simedges.values()]
        if sum(ratios) == 0: continue
        ratios = [(idx, elem/sum(ratios)) for idx, elem in enumerate(ratios)]
        ratios = sorted(ratios, key=lambda x:x[1], reverse=True)
        
        classids = []
        acc = 0
        idx, far = 0, 0
        while idx < len(ratios):
            if ratios[idx][1] == 0: break
            if acc >= max_ratio: break
            if len(classids) == n_component: break

            far = idx+1
            while far < len(ratios) and abs(ratios[far][1]-ratios[idx][1]) < 1e-5:
                far += 1
            
            candidate_ids = ratios[idx:far]
            n_remain = n_component - len(classids)
            # logger.info(f"{candidate_ids}, {len(classids)}, {n_component}, {n_remain}")
            if n_remain < len(candidate_ids):
                partids = random.sample(candidate_ids, k=n_remain)
            else:
                partids = candidate_ids
            classids.extend([elem[0] for elem in partids])
            acc += sum([elem[1] for elem in partids])
            idx = far
                    
        tagid2classids[tag] = classids
    return tagid2classids

def merge_simedges_from_similar_tags(tag2simedges_mp:dict, classid2tagids:dict,):
    classid2simedges = {}
    for class_id, tagids in classid2tagids.items():
        cnt = sum([len(tag2simedges_mp[tagid][class_id]) for tagid in tagids])
        logger.info(f"{class_id:>2}, {cnt:>10}")
        simedges = []
        for tagid in tagids:
            simedges.extend(tag2simedges_mp[tagid][class_id])
        classid2simedges[class_id] = simedges
    return classid2simedges

def merge_user_edges_with_topic_edges(user_edges, classid2simedges:dict, user_size:int, add_self_loop=False,):
    user_edges = user_edges + [(i,i) for i in range(user_size)]

    classid2simmat = {}
    for key, simedges in classid2simedges.items():
        edges = list(zip(*user_edges+simedges))
        edges_t = torch.LongTensor(edges) # (2,#num_edges)
        weight_t = torch.FloatTensor([1]*edges_t.size(1))
        classid2simmat[key] = Data(edge_index=edges_t, edge_weight=weight_t)
    
    return classid2simmat

def buildMotifInducedAdjacencyMatrix(relation_matrix, user_size):
    # build csr matrix
    row, col = relation_matrix.edge_index
    weight = relation_matrix.edge_weight
    self_loop = np.arange(user_size)
    row = np.concatenate([row, self_loop])
    col = np.concatenate([col, self_loop])
    weight = np.concatenate([weight, np.ones(user_size)])
    S = coo_matrix((weight.tolist(), (row.tolist(),col.tolist())), shape=(user_size, user_size), dtype=np.float16)
    
    B = S.multiply(S.transpose())
    U = S - B
    C1 = (U.dot(U)).multiply(U.transpose())
    A1 = C1 + C1.transpose()
    C2 = (B.dot(U)).multiply(U.transpose()) + (U.dot(B)).multiply(U.transpose()) + (U.dot(U)).multiply(B)
    A2 = C2 + C2.transpose()
    C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
    A3 = C3 + C3.transpose()
    A4 = (B.dot(B)).multiply(B)
    C5 = (U.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(U) + (U.transpose().dot(U)).multiply(U)
    A5 = C5 + C5.transpose()
    A6 = (U.dot(B)).multiply(U) + (B.dot(U.transpose())).multiply(U.transpose()) + (U.transpose().dot(U)).multiply(B)
    A7 = (U.transpose().dot(B)).multiply(U.transpose()) + (B.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(B)
    A = S + A1 + A2 + A3 + A4 + A5 + A6 + A7
    A = coo_matrix((weight, (row, col)), shape=(user_size,user_size), dtype=np.float16)
    A = A.transpose().multiply(1.0/A.sum(axis=1).reshape(1, -1))
    A = A.transpose()
    return A

def build_heteredge_mats(data_dict:list, window_size:int, n_component:int,):
    # 1. Build Tag2SimEdges Mp
    tag2simedges_mp_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/tweet-embedding/llm-topic/tag2simedges_mp_windowsize{window_size}_model_tweet-topic-21-multi.pkl")
    if os.path.exists(tag2simedges_mp_filepath):
        tag2simedges_mp = load_pickle(tag2simedges_mp_filepath)
    else:
        # NOTE: we use timelines instead of preprocess_timelines bcz of aligning timelines and labels_aggby_timeline
        tag2simedges_mp = build_sim_edges(data_dict=data_dict, window_size=window_size)
        save_pickle(tag2simedges_mp, tag2simedges_mp_filepath)
    
    # 2. Reduce Timelines With Users WithIn Preprocess Timelines
    # tag2simedges_mp = {key:value for key,value in tag2simedges_mp.items() if key in preprocess_timelines_keys}
    # assert len(tag2simedges_mp) == len(preprocess_timelines_keys)

    # Adjust Unbalanced Labels in Class 12
    tag2toomuchedges = {}
    for tag,simedges in tag2simedges_mp.items():
        tag2toomuchedges[tag] = random.sample(simedges[12], k=int(0.01*len(simedges[12])))
    
    tag2simedges_unbalanced = {}
    for tag, simedges in tag2simedges_mp.items():
        simedges[12] = tag2toomuchedges[tag]
        tag2simedges_unbalanced[tag] = simedges
    
    # 3. Build Class2SimEdges Mp & Tag2Classid Mp
    tagid2classids = find_prominent_components_foreach_tag(tag2simedges_unbalanced, n_component=n_component)
    save_pickle(tagid2classids, os.path.join(DATA_ROOTPATH, f"Weibo-Aminer/llm/tagid2classids_windowsize{window_size}.pkl"))

    classid2tagids = {}
    for tag, classids in tagid2classids.items():
        for class_id in classids:
            if class_id not in classid2tagids:
                classid2tagids[class_id] = []
            classid2tagids[class_id].append(tag)
    
    classid2simedges = merge_simedges_from_similar_tags(tag2simedges_unbalanced, classid2tagids)
    save_pickle(classid2simedges, os.path.join(DATA_ROOTPATH, f"Weibo-Aminer/llm/classid2simedges_windowsize{window_size}.pkl"))

    # # 4. Build Class2Mat Mp
    # classid2simmat = {}
    # for class_id, simedges in classid2simedges.items():
    #     extend_adj = create_sparsemat_from_edgelist(user_edges+simedges, m=n_user, n=n_user)
    #     extend_adj = get_sparse_tensor(extend_adj.tocoo())
    #     classid2simmat[class_id] = extend_adj

    # classid2simmat = {}
    # for class_id, simedges in classid2simedges.items():
    #     edges = list(zip(*user_edges+simedges))
    #     edges_t = torch.LongTensor(edges) # (2,#num_edges)
    #     weight_t = torch.FloatTensor([1]*edges_t.size(1))
    #     classid2simmat[class_id] = Data(edge_index=edges_t, edge_weight=weight_t)

    return classid2simedges, tagid2classids

def build_heteredge_mats2(data_dict:dict, _u2idx:dict, window_size:int, n_component:int, dataset:str='Weibo-Aminer'):
    t_cascades = {}
    for tag, cascades in data_dict.items():
        userlist = [[_u2idx[user], ts, label] for user, ts, label in zip(cascades['user'], cascades['ts'], cascades['label']) if user in _u2idx]
        pair_user = []
        full_size = min(window_size, len(userlist))
        for size in range(full_size):
            pair_user.extend([(i[0], j[0], j[1], i[2], j[2], i[2]==j[2]) for i, j in zip(userlist[::1], userlist[size::1])])
        t_cascades[tag] = pair_user
    
    t_cascades_pd = pd.DataFrame([elem for value in t_cascades.values() for elem in value])
    t_cascades_pd.columns = ["user1", "user2", "timestamp", 'label1', 'label2', 'same_label']
    max_label = max(t_cascades_pd[['label1','label2']].max().tolist())

    # build tag2classids
    tag2simedges = {}
    for tag, pair_users in t_cascades.items():
        simedges = {}
        for topic_i in range(max_label+1):
            if topic_i == 12: pair_users = random.choices(pair_users, k=int(0.1*len(pair_users)))
            simedges[topic_i] = list(filter(lambda x:x[3]==topic_i and x[5]==True, pair_users))
        tag2simedges[tag] = simedges
    tagid2classids = find_prominent_components_foreach_tag(tag2simedges, n_component=n_component)
    save_pickle(tagid2classids, os.path.join(DATA_ROOTPATH, f"{dataset}/llm/tagid2classids_windowsize{window_size}.pkl"))

    # build class2simedges
    t_cascades_pd = t_cascades_pd.sort_values(by="timestamp")
    classid2simedges = dict()
    for topic_i in range(max_label+1):
        t_cascades_pd_sub = t_cascades_pd
        t_cascades_pd_sub = t_cascades_pd_sub[(t_cascades_pd_sub['same_label']==True) & (t_cascades_pd_sub['label1'] == topic_i)]
        if topic_i == 12: t_cascades_pd_sub = t_cascades_pd_sub.sample(frac=0.1, replace=False, random_state=2023)
        classid2simedges[topic_i] = t_cascades_pd_sub.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
        # logger.info(f"{topic_i}, {len(classid2simedges[topic_i])}")
    save_pickle(classid2simedges, os.path.join(DATA_ROOTPATH, f"{dataset}/llm/classid2simedges_windowsize{window_size}.pkl"))

    user_edges = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/{dataset}/edges.data")
    classid2simmat = merge_user_edges_with_topic_edges(user_edges, classid2simedges, user_size=len(_u2idx), add_self_loop=True)
    save_pickle(classid2simmat, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/{dataset}/topic_diffusion_graph_windowsize{window_size}.data")

    classid2simmat2 = {}
    for classid, simmat in classid2simmat.items():
        motif_adj = buildMotifInducedAdjacencyMatrix(simmat, len(_u2idx))
        edge_index, edge_weight = from_scipy_sparse_matrix(motif_adj)
        classid2simmat2[classid] = Data(edge_index=edge_index, edge_weight=edge_weight)
        # logger.info(f"{classid}, {classid2simmat2[classid].edge_index.size(1)}")
    save_pickle(classid2simmat2, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/{dataset}/topic_diffusion_motif_graph_windowsize{window_size}.data")

    return classid2simmat2, tagid2classids
