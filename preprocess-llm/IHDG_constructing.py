import pickle
import numpy as np
# import pandas as pd
import os
import re
from openai import OpenAI
import httpx
import json
import random
import regex
import glob
import requests
# from bs4 import BeautifulSoup
from typing import Optional
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import datetime
import logging

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter

def save_pickle(obj, filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p','.data']:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    elif ext == '.npy':
        if not isinstance(obj, np.ndarray):
            obj = np.array(obj)
        np.save(filename, obj)
    else:
        pass # raise Error

def load_pickle(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p','.data']:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    elif ext == '.npy':
        return np.load(filename)
    else:
        return None # raise Error

def analyse_distribution(data):
    for i in range(10):
        print(np.percentile(data, i*10))


# 4. generate graph

# 4-1. Prepare input from TopicGPT
result_filepath = "/topicGPT/data/output/Twitter-Huangxin/result_twitter.jsonl"
regex_pattern = r'^\[1\] (.+?): (.+?) \| Confidence: (\d+)'

miss_cnt = 0
results = []

with open(result_filepath, 'r', encoding='utf-8') as f:
    for line in f:
        # resp = line
        resp = json.loads(line)
        # print(json.dumps(resp, indent=4, ensure_ascii=False))

        match = re.match(regex_pattern, resp['response'])
        if match:
            category = match.group(1)
            # description = match.group(2)
            # quote = match.group(3)
            confidence = int(match.group(3))

            # NOTE: confidence is not used
            if confidence >= 6:
                results.append([category, int(resp['id']), int(resp['ts'])])
            else:
                miss_cnt += 1 
        else:
            miss_cnt += 1
            logger.info(resp)
logger.info(f"Results count: {len(results)}, Miss count: {miss_cnt}")

# 4-1-2. Prepare topic to interest mappings
gt_ft_mapping = load_pickle("/data/Twitter-Huangxin/output/gt_ft_mapping.data")

# 4-2. generate simedges from interest-aware cascades
def generate_interest_cascades_nomapping(data_dict: list):
    interest_cascades = {}
    for elem in data_dict:
        if elem[0] not in interest_cascades: interest_cascades[elem[0]] = []
        interest_cascades[elem[0]].append((elem[1], int(elem[2])))
    logger.info(f"1-interest_cascades: {len(interest_cascades)}")

    # sort by timestamp
    for t in interest_cascades:
        interest_cascades[t] = sorted(interest_cascades[t], key=lambda x: x[1])
    return interest_cascades

def generate_interest_simedges(interest_cascades: dict, time_distance: int = 3600 * 24 * 30):
    simedges = {}
    for interest, cascades in interest_cascades.items():
        if gt_ft_mapping[interest] not in simedges:
            simedges[gt_ft_mapping[interest]] = []
        for i in range(len(cascades)-1):
            for j in range(i, len(cascades)-1):
                if cascades[j][1] - cascades[i][1] < time_distance and \
                    cascades[j][0] != cascades[i][0]:
                    simedges[gt_ft_mapping[interest]].append((cascades[i][0], cascades[j][0]))

    # remove abundant edges
    for interest, edges in simedges.items():
        simedges[interest] = list(set(edges))

    # remove minor interests
    simedges_cp = simedges.copy()
    for t, cascades in simedges_cp.items():
        if len(cascades) < 10:
            simedges.pop(t)
    
    logger.info(f"simedges: {len(simedges)}")
    logger.info("Keys in simedges:")
    for key in simedges.keys():
        logger.info(key)

    return simedges

# 4-3. convert simedges to graph
def convert_to_graph(simedges: dict, user_size: int, originial_edges: Optional[list] = None):
    if originial_edges:
        for interest, edges in simedges.items():
            simedges[interest] = list(set(edges + originial_edges))
    
    # add self-loop edges
    for interest, edges in simedges.items():
        edges += [(u,u) for u in range(user_size)]
        simedges[interest] = list(set(edges))
    
    # convert to graph
    graph_d = {}
    for interest, edges in simedges.items():
        edges = list(zip(*edges))
        edges_t = torch.LongTensor(edges)
        weight_t = torch.FloatTensor([1]*edges_t.size(1))
        graph_d[interest] = Data(edge_index=edges_t, edge_weight=weight_t)

        graph_adj = to_dense_adj(edge_index=edges_t, edge_attr=weight_t).squeeze()
        graph_adj[graph_adj!=0] = 1.

    logger.info("Keys in graph_d:")
    for key in graph_d.keys():
        logger.info(key)
    
    return graph_d

def sort_dict(m: dict) -> dict:
    return dict(sorted(m.items(), key=lambda x: x[1], reverse=True))

days = 7
time_distance = 3600 * 24 * days
save_filepath = "/data/Twitter-Huangxin/output/new_topic_diffusion_graph_full_windowsize{td}.data"
save_filepath = save_filepath.format(td=days)
logger.info(save_filepath)

interest_cascades_nomapping = generate_interest_cascades_nomapping(results)
simedges = generate_interest_simedges(interest_cascades_nomapping, time_distance)

u2idx = load_pickle("/data/Twitter-Huangxin/u2idx.data")
original_edges = load_pickle("/data/Twitter-Huangxin/edges.data")
new_idx2u = load_pickle("/data/Twitter-Huangxin/idx2u.data")
max_user_size = new_idx2u[-1] + 1
logger.info(f"max_user_size: {max_user_size}")
graph_d = convert_to_graph(simedges, user_size=max_user_size, originial_edges=original_edges)
for key, graph in graph_d.items():
    logger.info(graph.edge_index.size())
save_pickle(graph_d, save_filepath)
print("The file has saved.")