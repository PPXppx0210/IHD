from datetime import datetime
from utils.log import logger
from utils.utils import *

def read_static_network():
    # Weibo Network: Static Following Network
    weibo_network_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/weibo_network.txt")

    edges = []
    with open(weibo_network_filepath, 'r') as f:
        n_users, n_edges = 0, 0
        for idx, line in enumerate(f):
            parts = line[:-1].split('\t')
            if idx == 0: 
                n_users, n_edges = int(parts[0]), int(parts[1])
                logger.info(f"{n_users}, {n_edges}")
            else:
                uid1, n_followees = parts[0], parts[1]
                # logger.info(f"{uid1}, {n_followees}")
                for uid2, relation in zip(parts[2::2], parts[3::2]):
                    # logger.info(f"{uid2},{relation}")
                    edges.append((uid1, uid2))

    logger.info(len(edges))
    return edges

def get_static_subnetwork(user_ids:set):
    # Weibo Network: Static Following Network
    weibo_network_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/weibo_network.txt")

    edges = []
    with open(weibo_network_filepath, 'r') as f:
        n_users, n_edges = 0, 0
        for idx, line in enumerate(f):
            parts = line[:-1].split('\t')
            if idx == 0: 
                n_users, n_edges = int(parts[0]), int(parts[1])
                logger.info(f"{n_users}, {n_edges}")
            else:
                uid1, n_followees = int(parts[0]), parts[1]
                if uid1 not in user_ids: continue
                # logger.info(f"{uid1}, {n_followees}")
                for uid2, relation in zip(parts[2::2], parts[3::2]):
                    # logger.info(f"{uid2},{relation}")
                    uid2, relation = int(uid2), int(relation)
                    if uid2 not in user_ids: continue
                    edges.append((uid1, uid2))
                    if relation == 1:
                        edges.append((uid2,uid1))
    
    logger.info(len(edges))
    return edges

def reindex_edges(nodes, edges):
    new_nodes = {}
    for idx, node in enumerate(nodes):
        new_nodes[node] = idx
    
    new_edges = []
    for u1, u2 in edges:
        if u1 in new_nodes and u2 in new_nodes:
            new_edges.append((new_nodes[u1], new_nodes[u2]))
    return new_nodes, new_edges

def read_uid_mid_mp():
    new2old_uid_mp = {}
    old2new_uid_mp = {}
    with open(os.path.join(DATA_ROOTPATH, "Weibo-Aminer/uidlist.txt"), 'r') as f:
        for idx, line in enumerate(f):
            new2old_uid_mp[idx] = int(line)
            old2new_uid_mp[int(line)] = idx
    logger.info(len(old2new_uid_mp))

    new2old_mid_mp = {}
    old2new_mid_mp = {}
    with open(os.path.join(DATA_ROOTPATH, "Weibo-Aminer/diffusion/repost_idlist.txt"), 'r') as f:
        for idx, line in enumerate(f):
            new2old_mid_mp[idx] = int(line)
            old2new_mid_mp[int(line)] = idx
    logger.info(len(old2new_mid_mp))

    return old2new_uid_mp, old2new_mid_mp

def read_diffusion_nocontent():
    diffusion_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/diffusion/repost_data.txt")
    # diffusion_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/diffusion/repost_data_sample.txt")

    diffusion = {}
    with open(diffusion_filepath, 'r') as f:
        cnt, mid = 0, None
        for line in f:
            parts = line.split('\t')
            if cnt == 0:
                # mid, cnt = old2new_mid_mp[int(parts[0])], int(parts[1])
                mid, cnt = int(parts[0]), int(parts[1])
                diffusion[mid] = []
            else:
                ts, uid = int(parts[0]), int(parts[1])
                # diffusion[mid].append((old2new_uid_mp[uid], ts))
                diffusion[mid].append((uid, ts))
                cnt -= 1

    save_pickle(diffusion, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/diffusion_withoutcontent_newids.pkl"))
    logger.info(len(diffusion))
    return diffusion

def read_diffusion_withcontent(old2new_uid_mp, old2new_mid_mp):
    retweet_content_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/weibocontents/Retweet_Content.txt")

    diffusion_withcontent = {}
    with open(retweet_content_filepath, 'r', encoding='gbk') as f:
        cnt, mid, line_id = 0, -1, 1
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            # logger.info(f"line_id={line_id}, mid={mid}, cnt={cnt}, parts={parts}")
            if len(parts) >= 1:
                elems = parts[0].split(' ')
                if elems[0] == 'retweet' or elems[0] == 'link' or elems[0] == '@': continue

            if cnt == 0:
                if len(parts) > 1:
                    mid = int(parts[0])
                    diffusion_withcontent[mid] = []
                else:
                    cnt = int(parts[0])
            else:
                if line_id == 1:
                    diffusion_withcontent[mid].append({
                        'uid': int(parts[0]), 'old_mid': int(parts[2]), 'ts': int(datetime.timestamp(datetime.strptime(parts[1], "%Y-%m-%d-%H:%M:%S"))),
                    })
                    line_id = 2
                elif line_id == 2:
                    elem = diffusion_withcontent[mid][-1]
                    elem['content'] = parts
                    line_id = 1
                    cnt -= 1

    diffusion_withcontent2 = {}
    for key, value in diffusion_withcontent.items():
        us = set()
        cascade = []
        for elem in value:
            if elem['uid'] in us: continue
            us.add(elem['uid'])
            # cascade.append({'uid': elem['uid'], 'mid': elem['mid'], 'ts': elem['ts'], 'content': elem['content']})
            cascade.append(elem)
        cascade = sorted(cascade, key=lambda elem:elem['ts'])
        diffusion_withcontent2[key] = cascade

    save_pickle(diffusion_withcontent2, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/diffusion_withcontent.pkl"))
    logger.info(len(diffusion_withcontent2))

    diffusion_withcontent3 = {}
    for tag, cascades in diffusion_withcontent2.items():
        if tag not in old2new_mid_mp: continue
        cascades_new = []
        for elem in cascades:
            if elem['uid'] not in old2new_uid_mp: continue
            cascades_new.append({'uid': old2new_uid_mp[elem['uid']], 'old_mid': elem['old_mid'], 'ts': elem['ts'], 'content': elem['content']})

        diffusion_withcontent3[old2new_mid_mp[tag]] = cascades_new

    save_pickle(diffusion_withcontent3, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/diffusion_withcontent_newids.pkl"))
    logger.info(len(diffusion_withcontent3))
    return diffusion_withcontent3

def read_originial_content(old2new_mid_mp):
    root_content_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/weibocontents/Root_Content.txt")
    midwithcontent = {}
    with open(root_content_filepath, 'r', encoding='gbk') as f:
        mid = -1
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                elems = parts[0].split(' ')
                if elems[0] == 'retweet' or elems[0] == 'link' or elems[0] == '@': continue
                if mid == -1:
                    assert len(elems) == 1
                    assert int(elems[0]) in old2new_mid_mp
                    mid = old2new_mid_mp[int(elems[0])]
                else:
                    midwithcontent[mid] = [int(word) if word != '' else word for word in elems[0].strip().split(' ')]
                    mid = -1

    save_pickle(midwithcontent, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/diffusion_original_content.pkl"))
    logger.info(len(midwithcontent))
    return midwithcontent

def read_user_ids(train_data_dict_filepath, valid_data_dict_filepath, test_data_dict_filepath, start_uid=0):
    train_data_dict = load_pickle(train_data_dict_filepath)
    valid_data_dict = load_pickle(valid_data_dict_filepath)
    test_data_dict  = load_pickle(test_data_dict_filepath)
    data_dict = {**train_data_dict, **valid_data_dict, **test_data_dict}

    # dict_keys = set(train_data_dict.keys()) | set(valid_data_dict.keys()) | set(test_data_dict)
    # dict_keys = [int(elem) for elem in dict_keys]

    cnt = start_uid
    uid_mp = {}
    key = 'user' if 'user' in list(data_dict.values())[0] else 'seq'
    for elem in data_dict.values():
        for uid in elem[key]:
            if uid in uid_mp: continue
            uid_mp[uid] = cnt
            cnt += 1
    
    return uid_mp

def read_wordtable():
    wordtable_filepath = os.path.join(DATA_ROOTPATH, "Weibo-Aminer/weibocontents/WordTable.txt")
    wordtable = {}
    num_words = -1
    with open(wordtable_filepath, 'r', encoding='gbk') as f:
        for idx, line in enumerate(f):
            if idx == 0: num_words = int(line)
            else:
                parts = line.strip().split('\t')
                wordtable[int(parts[0])] = {
                    'cnt': int(parts[1]),
                    'word': parts[2]
                }

    logger.info(f"{len(wordtable)}, {num_words}")
    return wordtable, num_words

def get_shorten_cascades_nocontent(diffusion_nocontent, us):
    # diffusion_nocontent = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/diffusion_withoutcontent_newids.pkl")
    # us = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/user_ids.pkl")

    shorten_diffusion_nocontent = {}
    for tag, cascades in diffusion_nocontent.items():
        shorten_cascades = []
        for elem in cascades:
            if elem[0] not in us: continue
            shorten_cascades.append(elem)
        if len(shorten_cascades) < 4: continue
        shorten_diffusion_nocontent[tag] = shorten_cascades

    save_pickle(shorten_diffusion_nocontent, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/shorten_diffusion_nocontent.pkl"))
    logger.info(len(shorten_diffusion_nocontent))
    return shorten_diffusion_nocontent

def get_shorten_cascades_withcontent(diffusion_withcontent, us):
    # diffusion_withcontent = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/diffusion_withcontent_newids.pkl")
    # us = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/user_ids.pkl")

    shorten_cascades_withcontent = {}
    for tag, cascades in diffusion_withcontent.items():
        shorten_cascades = []
        for elem in cascades:
            if elem['uid'] not in us: continue
            shorten_cascades.append(elem)
        if len(shorten_cascades) < 4: continue

        for elem in shorten_cascades:
            elem['content'] = [int(word) if word != '' else word for word in elem['content'][0].strip().split(' ')]
        shorten_cascades_withcontent[tag] = shorten_cascades

    save_pickle(shorten_cascades_withcontent, os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/shorten_diffusion_withcontent.pkl"))
    logger.info(len(shorten_cascades_withcontent))
    return shorten_cascades_withcontent

def select_and_merge_cascades(data_dict_keys, shorten_cascades_withcontent, root_content, wordtable):
    data_dict_withcontent = {}
    for key in data_dict_keys:
        if type(key) != 'int': key = int(key)
        users, tss, contents, words = [], [], [], []
        for elem in shorten_cascades_withcontent[key]:
            content = root_content[key].copy()
            if elem['content'] != ['']:
                content += elem['content']
            word = "".join([wordtable[word]['word'] for word in content if word in wordtable])
            users.append(elem['uid']); tss.append(elem['ts']); contents.append(content); words.append(word)
        data_dict_withcontent[key] = {
            'user': users,
            'ts': tss,
            'content': contents,
            'word': words,
        }
    return data_dict_withcontent
