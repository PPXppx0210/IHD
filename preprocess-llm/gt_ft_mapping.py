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


### start of 1 ###
result_filepath = "/topicGPT/data/output/twitter/result_twitter.jsonl"

regex_pattern = r'^(.+?): (.+?) \| Confidence: (\d+)'

miss_cnt = 0
generated_topics = {}

with open(result_filepath, 'r') as f:
    for line in f:
        # resp = line
        resp = json.loads(line)
        # print(json.dumps(resp, indent=4, ensure_ascii=False))

        match = re.match(regex_pattern, resp['response'])
        if match:
            category = match.group(1)
            # description = match.group(2)
            confidence = match.group(3)

            if category not in generated_topics: generated_topics[category] = {"count": 0}
            generated_topics[category]["count"] += 1
        else:
            miss_cnt += 1

print(len(generated_topics))
print(sum([v["count"] for k,v in generated_topics.items()]))

wiki_kb_filepath = "/preprocess-llm/base/wikipedia_class.txt"

first_topics = []
second_topics = []
st_ft_mapping = {}
last_ft = None
with open(wiki_kb_filepath, 'r') as f:
    for line in f:
        if line[0] == ' ':
            st_ft_mapping[line.strip()] = last_ft
            second_topics.append(line.strip())
        else:
            last_ft = line.strip()
            first_topics.append(line.strip())

# add first topics
for ft in first_topics:
    st_ft_mapping[ft] = ft

# print(first_topics.keys())

# 2. Prepare sBert encoding
from sentence_transformers import SentenceTransformer, util

sbert = SentenceTransformer("/all-MiniLM-L6-v2")

f_topic_mp = {}
f_topic_embs = sbert.encode(first_topics)
for tp, emb in zip(first_topics, f_topic_embs):
    f_topic_mp[tp] = emb

s_topic_mp = {}
s_topic_embs = sbert.encode(second_topics)
for tp, emb in zip(second_topics, s_topic_embs):
    s_topic_mp[tp] = emb

# 3. Find the most related wikipedia topics for TopicGPT-generated results
gts = list(generated_topics.keys())

gt_ft_mapping = {}
gt_embs = sbert.encode(gts)
for emb, tp in zip(gt_embs, gts):
    max_sim = -1
    max_tp = ""
    for f_tp, f_emb in f_topic_mp.items():
        sim = util.pytorch_cos_sim(emb, f_emb)
        if sim > max_sim:
            max_sim = sim
            max_tp = f_tp

    for s_tp, s_emb in s_topic_mp.items():
        sim = util.pytorch_cos_sim(emb, s_emb)
        if sim > max_sim:
            max_sim = sim
            max_tp = s_tp
    gt_ft_mapping[tp] = st_ft_mapping[max_tp]

save_pickle(gt_ft_mapping, "/data/Twitter-Huangxin/output/gt_ft_mapping.data")
