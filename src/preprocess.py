# Aminer Dataset
from utils.graph_aminer import *
from utils.graph import init_graph, build_heteredge_mats, build_heteredge_mats2
from utils.utils import split_cascades, load_pickle, save_pickle
from utils.graph_aminer import *
from utils.tweet_embedding import agg_tagemb_by_user
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from scipy.special import expit
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data
import igraph

# NOTE:
# Preliminaries: {train,valid,test}.data, edges.data
# Output: {train,valid,test}_withlabel_withcontent.data, classid2simmat.pkl(multi-heter-graph), diffusion_graph(*)
# Output2: deepwalk_feat.npy, three-sort_feat.npy, structural_feat.npy, tweet-agg_feat.npy

# For Twitter Dataset, we also split cascades into {train/valid/test}.data first

user_ids = read_user_ids()
user_ids = user_ids.values()
edges = get_static_subnetwork(user_ids)
_, new_edges = reindex_edges(user_ids, edges)
graph = init_graph(len(user_ids), new_edges, True, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre", True)

old2new_uid_mp, old2new_mid_mp = read_uid_mid_mp()

# diffusion = read_diffusion_nocontent()
# shorten_diffusion_nocontent = get_shorten_cascades_nocontent(diffusion, user_ids)
diffusion_withcontent = read_diffusion_withcontent(old2new_uid_mp, old2new_mid_mp)
shorten_cascades_withcontent = get_shorten_cascades_withcontent(diffusion_withcontent, user_ids)

midcontent = read_originial_content(old2new_mid_mp)
wordtable = read_wordtable()

# train_dict_keys, valid_dict_keys, test_dict_keys = split_cascades(shorten_cascades_withcontent, train_ratio=0.8, valid_ratio=0.1)

with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train.data", 'rb') as file:
    train_data_dict = pickle.load(file)

train_data_dict_withcontent = select_and_merge_cascades(train_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
save_pickle(train_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent.pkl")

with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid.data", 'rb') as file:
    valid_data_dict = pickle.load(file)

valid_data_dict_withcontent = select_and_merge_cascades(valid_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
save_pickle(valid_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid_withcontent.pkl")

with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test.data", 'rb') as file:
    test_data_dict = pickle.load(file)

test_data_dict_withcontent = select_and_merge_cascades(test_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
save_pickle(test_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test_withcontent.pkl")

# NOTE: bertopic-preprocess/{llm-normtext, llm-topic, llm-tag}.py

train_data_dict_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent.pkl"
valid_data_dict_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid_withcontent.pkl"
test_data_dict_filepath  = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test_withcontent.pkl"

MODEL = f"cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
class_mapping = model.config.id2label
if torch.cuda.is_available():
    model = model.to('cuda')

def get_topic_label(data_dict, model, tokenizer):
    for tag, cascades in tqdm(data_dict.items()):
        words = cascades['word']
        with torch.no_grad():
            tokens = tokenizer(words, return_tensors='pt', padding=True)
            if torch.cuda.is_available():
                tokens = {
                    'input_ids': tokens['input_ids'].to('cuda'),
                    # 'token_type_ids': tokens['token_type_ids'].to('cuda'),
                    'attention_mask': tokens['attention_mask'].to('cuda'),
                }
            output = model(**tokens)
            embeds = output.logits.detach().cpu()
            scores = expit(embeds)
            labels = np.array([np.where(score==max(score))[0][0] for score in scores])
        cascades['label'] = labels

        del tokens, output
        torch.cuda.empty_cache()
    return data_dict

train_data_dict = load_pickle(train_data_dict_filepath)
train_data_dict_e = get_topic_label(train_data_dict)
save_pickle(train_data_dict_e, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent_withlabel.pkl")

valid_data_dict = load_pickle(valid_data_dict_filepath)
valid_data_dict_e = get_topic_label(valid_data_dict)
save_pickle(valid_data_dict_e, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid_withcontent_withlabel.pkl")

test_data_dict = load_pickle(test_data_dict_filepath)
test_data_dict_e = get_topic_label(test_data_dict)
save_pickle(test_data_dict_e, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test_withcontent_withlabel.pkl")

data_dict = {**train_data_dict, **valid_data_dict, **test_data_dict}

user_set = read_user_ids(train_data_dict_filepath, valid_data_dict_filepath, test_data_dict_filepath)
midwithcontent = load_pickle(os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/diffusion_original_content.pkl"))
wordtable = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/Aminer-pre/wordtable.pkl")

# NOTE: Build HeterEdge Mats
user_edges = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/edges.pkl")

classid2simedges, tagid2classids = build_heteredge_mats(data_dict=load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent.pkl"), window_size=200, n_component=3)
classid2newsimedges = {}
u2idx = old2new_uid_mp
for classid, simedges in classid2simedges.items():
    new_simedges = []
    for uid1, uid2 in simedges:
        if uid1 in u2idx and uid2 in u2idx:
            new_simedges.append((u2idx[uid1], u2idx[uid2]))
    logger.info(f"{classid}, {len(new_simedges)}, {len(simedges)}")
    classid2newsimedges[classid] = new_simedges

classid2simmat = {}
for class_id, simedges in classid2newsimedges.items():
    edges = list(zip(*user_edges+simedges))
    edges_t = torch.LongTensor(edges) # (2,#num_edges)
    weight_t = torch.FloatTensor([1]*edges_t.size(1))
    classid2simmat[class_id] = Data(edge_index=edges_t, edge_weight=weight_t)
save_pickle(classid2simmat, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/llm/classid2simmat_windowsize200.pkl")

# NOTE: User-Side Feats
# structural_feat = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/vertex_feature_subgle483.npy"))
# three_sort_feat = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/user_features_avg.p"))
# deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deepwalk/deepwalk_added.emb_64"), max_idx=user_nodes[-1]+1)

pretrained_model_name = 'xlm-roberta-base'
user2emb = agg_tagemb_by_user(n_user=len(user_set), cascades=data_dict, pretrained_model_name='xlm-roberta-base')
save_pickle(user2emb, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/llm/tag_embs_aggbyuser_model_xlm-roberta-base_pca_dim128.pkl")

# Basic
edgelist = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/edges.data")
u2idx = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/u2idx.data")

graph = igraph.Graph(len(u2idx), directed=True)
graph.add_edges(edgelist)
graph.simplify()
save_pickle(graph, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/basic/graph.pkl")

cascades = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/cascades.data")
u2idx = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/u2idx.data")
new_cascades = {}
for key, cascade in cascades.items():
    new_cascades[key] = [(u2idx[a],b) for a,b in zip(cascade['user'],cascade['ts'])]
save_pickle(new_cascades, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/basic/timeline.pkl")

# Statistics
graph:igraph.Graph = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/basic/graph.pkl")
deg = graph.degree()
# deg = list(filter(lambda x:x>97, deg))
summarize_distribution(deg)
# NOTE: 85% -> 90% -> 95% -> filter90%+50% -> filter90%+90%

timelines:Dict[int,list] = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/basic/timeline.pkl")
t_len = [len(cascade) for _, cascade in timelines.items()]
# t_len = list(filter(lambda x:x>102, t_len))
# NOTE: 5% -> 90% -> filter90%+50%

# Build Heter-Edge Graphs
n_component = 3
for window_size in [200, 300]:
    _, _ = build_heteredge_mats2(data_dict, u2idx, window_size=window_size, n_component=n_component, dataset='Weibo-Aminer')

# Cascade Triplet Txt
# cascade_dict = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/cascades.data")

# tag2idx = {}
# index = 0
# for tag, cascades in cascade_dict.items():
#     tag2idx[tag] = index
#     index += 1

# with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/cascade_triplet.txt", 'w') as f:
#     for tag, cascades in cascade_dict.items():
#         for user, ts in zip(cascades['user'], cascades['label']):
#             f.write(f"{ts} {tag2idx[tag]} {user}\n")

# Build Heter-Deepwalk(Node2Vec) Feats
edgelist_mp = {}

diffusion_graph = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_diffusion_graph_windowsize200.data")
for classid, mat in diffusion_graph.items():
    edgelist = []
    for from_, to_ in zip(mat.edge_index[0], mat.edge_index[1]):
        edgelist.append((from_, to_))
    edgelist_mp[classid] = edgelist

for classid, edgelist in edgelist_mp.items():
    with open(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_graph_edgelist/edgelist_ws200_topic{classid}.txt", 'w') as f:
        for from_, to_ in edgelist:
            f.write(f"{from_} {to_}\n")

# for i in `seq 1 17`
# do
#     # echo $i
#     deepwalk --input /remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_graph_edgelist/edgelist_ws200_topic${i}.txt --format edgelist --output /remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_graph_edgelist/feature/topicg_deepwalk_topic${i}.data > log-topicg-deepwalk-topic${i}.txt 2>&1
# done
