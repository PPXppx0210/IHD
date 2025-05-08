# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))
from utils.log import logger
# from lib.utils import get_node_types, extend_edges, get_sparse_tensor
import pickle
import numpy as np
# import pandas as pd
import os
import psutil
import subprocess
import random
import configparser
import torch
from scipy import sparse
from typing import Any, Dict, List
from itertools import combinations, chain
from sklearn.decomposition import PCA
# from numba import njit
# from numba import types
# from numba.typed import Dict, List

config = configparser.ConfigParser()
# NOTE: realpath(__file__)是在获取执行这段代码所属文件的绝对路径, 即~/pyHeter-GAT/src/config.ini
config.read(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/config.ini'))
DATA_ROOTPATH = config['DEFAULT']['DataRootPath']
Ntimestage = int(config['DEFAULT']['Ntimestage'])

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

def check_memusage_GB():
    return psutil.Process().memory_info().rss / (1024*1024*1024)

def check_gpu_memory_usage(gpu_id:int):
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[gpu_id]

def save_model(epoch, args, model, optimizer, filepath:str=None):
    state = {
        "epoch": epoch,
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.get_state_dict() if type(optimizer).__name__ == 'ScheduledOptim' else optimizer.state_dict()
    }
    save_filepath = filepath
    if save_filepath is None:
        save_filepath = os.path.join(DATA_ROOTPATH, f"basic/training/ckpt_epoch_{epoch}_model_{args.model}.pkl")
    torch.save(state, save_filepath)
    # help release GPU memory
    del state
    logger.info(f"Save State To Path={save_filepath}... Checkpoint Epoch={epoch}")

def load_model(filename, model):
    ckpt_state = torch.load(filename, map_location='cpu')
    model.load_state_dict(ckpt_state['model'])
    # optimizer.load_state_dict(ckpt_state['optimizer'])
    logger.info(f"Load State From Path={filename}... Checkpoint Epoch={ckpt_state['epoch']}")

def summarize_distribution(data: List[Any]):
    logger.info(f"list length={len(data)}")
    logger.info("mean list length %.2f", np.mean(data))
    logger.info("max list length %.2f", np.max(data))
    logger.info("min list length %.2f", np.min(data))
    for i in range(1, 20):
        logger.info("%d-th percentile of list length %.2f", i*5, np.percentile(data, i*5))

def wrap(func):
    """
    Usage: wrap(graph.degree)(0)
    """
    def inner(*args, **kwargs):
        # logger.info(f"func={func}, args={args}, kwargs={kwargs}")
        return func(*args, mode="out", **kwargs)
    return inner

def flattern(ll:List[list]):
    return [item for sublist in ll for item in sublist]

def normalize(feat:np.ndarray):
    return feat/(np.linalg.norm(feat,axis=1)+1e-10).reshape(-1,1)

def reduce_dimension(feats:np.ndarray, reduce_dim:int):
    feats = feats.transpose((1,0))
    pca = PCA(n_components=reduce_dim)
    pca.fit(feats)
    reduced_feats = pca.components_
    reduced_feats = reduced_feats.transpose((1,0))
    return reduced_feats

def load_w2v_feature(file, max_idx=0):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                # logger.info(f"n={n}, d={d}")
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            while len(feature) <= index:
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)

def save_w2v_feature(file, feature):
    n, d = feature.shape

    with open(file, "wb") as f:
        f.write(f"{n} {d}\n".encode())
        for i, row in enumerate(feature):
            line = ' '.join(str(num) for num in row)
            f.write(f"{i} {line}\n".encode())

def sample_docs_foreachuser(docs, user_tweet_mp, sample_frac=0.01, min_sample_num=3):
    sample_docs = []
    for user_id in range(len(user_tweet_mp)):
        docs_id = user_tweet_mp[user_id]
        sample_num = int(sample_frac*len(docs_id))
        user_docs = random.choices(docs[docs_id[0]:docs_id[-1]+1], k=max(sample_num, min_sample_num))
        sample_docs.extend(user_docs)
    logger.info(f"sample_docs_num={len(sample_docs)}")
    return sample_docs

def sample_docs_foreachuser2(docs, sample_frac=0.01, min_sample_num=3):
    """
    func: process docs with format of lists of lists, i.e. [[]]
    """
    sample_docs = []
    for texts in docs:
        if len(texts) == 0:
            continue
        texts = list(set(texts))
        sample_num = int(sample_frac*len(texts))
        sample_texts = random.choices(texts, k=max(sample_num, min_sample_num))
        sample_docs.extend(sample_texts)
    logger.info(f"sample_docs_num={len(sample_docs)}")
    return sample_docs

def split_cascades(cascades:dict, train_ratio=0.8, valid_ratio=0.1):
    keys = list(cascades.keys())
    random.shuffle(keys)
    train_dict_keys, valid_dict_keys, test_dict_keys = np.split(keys, [int(train_ratio*len(keys)), int((train_ratio+valid_ratio)*len(keys))])
    return train_dict_keys, valid_dict_keys, test_dict_keys
