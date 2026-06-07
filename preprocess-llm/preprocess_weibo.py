import argparse
import datetime
import json
import logging
import os
import pickle
import random
import re
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PREPROCESS_ROOT = os.path.dirname(os.path.realpath(__file__))
PPX_SHARE_ROOT = "/remote-home/share/dmb_nas/ppx"
DEFAULT_DATA_ROOT = os.environ.get("IHD_DATA_ROOT", os.path.join(PPX_SHARE_ROOT, "data"))
DEFAULT_WIKI_FILE = os.path.join(PREPROCESS_ROOT, "wikipedia_class.txt")
DEFAULT_LOCAL_SBERT_MODEL = os.path.join(PROJECT_ROOT, "all-MiniLM-L6-v2")
DEFAULT_SBERT_MODEL = os.environ.get(
    "SBERT_MODEL",
    DEFAULT_LOCAL_SBERT_MODEL if os.path.isdir(DEFAULT_LOCAL_SBERT_MODEL) else "all-MiniLM-L6-v2",
)
DATASET_NAME = "Weibo-Aminer"
GRAPH_PREFIX = "wb"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_SHUFFLE_RATIO = 0.0
RESPONSE_PATTERN = re.compile(
    r'^\s*\[1\]\s*(.+?):\s*(.*?)(?:\s*\(".*?"\))?\s*\|\s*Confidence:\s*(\d+)\s*$'
)


def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.Formatter.converter = Beijing_TimeZone_Converter


def save_pickle(obj, filename):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    _, ext = os.path.splitext(filename)
    if ext in [".pkl", ".p", ".data"]:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    elif ext == ".npy":
        if not isinstance(obj, np.ndarray):
            obj = np.array(obj)
        np.save(filename, obj)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


def load_pickle(filename):
    _, ext = os.path.splitext(filename)
    if ext in [".pkl", ".p", ".data"]:
        with open(filename, "rb") as f:
            return pickle.load(f)
    if ext == ".npy":
        return np.load(filename)
    raise ValueError(f"Unsupported input extension: {ext}")


def coerce_user_id(user_id):
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return user_id


def extract_user_ts(resp: dict):
    meta_info = resp.get("meta_info") or {}
    user_id = resp.get("id")
    if user_id is None:
        user_id = resp.get("user")
    if user_id is None:
        user_id = meta_info.get("user")
    if user_id is None:
        user_id = meta_info.get("id")

    timestamp = resp.get("ts")
    if timestamp is None:
        timestamp = resp.get("timestamp")
    if timestamp is None:
        timestamp = meta_info.get("ts")
    if timestamp is None:
        timestamp = meta_info.get("timestamp")

    if user_id is None or timestamp is None:
        raise KeyError("user id or timestamp is missing")
    return coerce_user_id(user_id), int(timestamp)


def lookup_user_index(u2idx: dict, user_id):
    if user_id in u2idx:
        return u2idx[user_id]

    user_str = str(user_id)
    if user_str in u2idx:
        return u2idx[user_str]

    try:
        user_int = int(user_id)
    except (TypeError, ValueError):
        return None
    return u2idx.get(user_int)


def parse_llm_results(result_filepath: str, confidence_threshold: int):
    results = []
    miss_cnt = 0
    with open(result_filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                miss_cnt += 1
                logger.warning("Line %s: JSON decode failed.", line_no)
                continue

            response_text = str(resp.get("response") or resp.get("responses") or "")
            match = RESPONSE_PATTERN.match(response_text)
            if not match:
                miss_cnt += 1
                logger.info("Line %s: response format mismatch: %s", line_no, response_text)
                continue

            try:
                confidence = int(match.group(3))
                if confidence >= confidence_threshold:
                    user_id, timestamp = extract_user_ts(resp)
                    results.append([match.group(1).strip(), user_id, timestamp])
                else:
                    miss_cnt += 1
            except (KeyError, TypeError, ValueError) as exc:
                miss_cnt += 1
                logger.warning("Line %s: id/ts/confidence parse failed: %s", line_no, exc)

    logger.info("Results count: %s, Miss count: %s", len(results), miss_cnt)
    return results


def collect_generated_topics(result_filepath: str):
    generated_topics = {}
    miss_cnt = 0
    with open(result_filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                resp = json.loads(line)
                response_text = str(resp.get("response") or resp.get("responses") or "")
            except json.JSONDecodeError:
                response_text = line.strip()

            match = RESPONSE_PATTERN.match(response_text)
            if not match:
                miss_cnt += 1
                logger.info("Line %s: response format mismatch while mapping: %s", line_no, response_text)
                continue

            category = match.group(1).strip()
            generated_topics.setdefault(category, {"count": 0})
            generated_topics[category]["count"] += 1

    logger.info(
        "Generated topics: %s, matched rows: %s, missed rows: %s",
        len(generated_topics),
        sum(v["count"] for v in generated_topics.values()),
        miss_cnt,
    )
    return generated_topics


def read_wikipedia_topics(wiki_kb_filepath: str):
    first_topics = []
    second_topics = []
    st_ft_mapping = {}
    last_ft = None

    with open(wiki_kb_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if line[0].isspace():
                if last_ft is None:
                    continue
                topic = line.strip()
                st_ft_mapping[topic] = last_ft
                second_topics.append(topic)
            else:
                last_ft = line.strip()
                first_topics.append(last_ft)

    for ft in first_topics:
        st_ft_mapping[ft] = ft

    logger.info("Wikipedia first topics: %s, second topics: %s", len(first_topics), len(second_topics))
    return first_topics, second_topics, st_ft_mapping


def build_topic_mapping(generated_topics: dict, wiki_kb_filepath: str, sbert_model: str):
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to build gt_ft_mapping_ppx.data. "
            "Install it on the remote server or pass an existing --mapping-file."
        ) from exc

    first_topics, second_topics, st_ft_mapping = read_wikipedia_topics(wiki_kb_filepath)
    gts = list(generated_topics.keys())
    if not gts:
        raise ValueError("No generated topics were parsed from the result file.")

    logger.info("Loading SentenceTransformer model: %s", sbert_model)
    sbert = SentenceTransformer(sbert_model)

    f_topic_embs = sbert.encode(first_topics)
    s_topic_embs = sbert.encode(second_topics)
    gt_embs = sbert.encode(gts)

    gt_ft_mapping = {}
    for emb, tp in zip(gt_embs, gts):
        max_sim = -1.0
        max_tp = ""
        for f_tp, f_emb in zip(first_topics, f_topic_embs):
            sim = float(util.pytorch_cos_sim(emb, f_emb).item())
            if sim > max_sim:
                max_sim = sim
                max_tp = f_tp
        for s_tp, s_emb in zip(second_topics, s_topic_embs):
            sim = float(util.pytorch_cos_sim(emb, s_emb).item())
            if sim > max_sim:
                max_sim = sim
                max_tp = s_tp
        gt_ft_mapping[tp] = st_ft_mapping[max_tp]

    return gt_ft_mapping


def load_or_build_topic_mapping(
    mapping_file: str,
    result_file: str,
    wiki_file: str,
    sbert_model: str,
    force_remap: bool = False,
):
    if os.path.exists(mapping_file) and not force_remap:
        logger.info("Loading existing topic mapping: %s", mapping_file)
        return load_pickle(mapping_file)

    logger.info("Building topic mapping: %s", mapping_file)
    generated_topics = collect_generated_topics(result_file)
    topic_mapping = build_topic_mapping(generated_topics, wiki_file, sbert_model)
    save_pickle(topic_mapping, mapping_file)
    logger.info("Saved topic mapping: %s (%s topics)", mapping_file, len(topic_mapping))
    return topic_mapping


def generate_interest_cascades_nomapping(data_dict: list):
    interest_cascades = {}
    for category, user_id, timestamp in data_dict:
        interest_cascades.setdefault(category, []).append((user_id, int(timestamp)))
    logger.info("1-interest_cascades: %s", len(interest_cascades))

    for interest in interest_cascades:
        interest_cascades[interest] = sorted(interest_cascades[interest], key=lambda x: x[1])
    return interest_cascades


def generate_interest_simedges(
    interest_cascades: dict,
    topic_mapping: dict,
    time_distance: int,
    min_edges: int,
    u2idx: Optional[dict] = None,
):
    simedges = {}
    missing_topics = set()
    for interest, cascades in interest_cascades.items():
        mapped_interest = topic_mapping.get(interest)
        if mapped_interest is None:
            missing_topics.add(interest)
            continue

        simedges.setdefault(mapped_interest, [])
        for i in range(len(cascades) - 1):
            for j in range(i, len(cascades) - 1):
                if cascades[j][1] - cascades[i][1] < time_distance and cascades[j][0] != cascades[i][0]:
                    u_i, u_j = cascades[i][0], cascades[j][0]
                    if u2idx is not None:
                        u_i, u_j = lookup_user_index(u2idx, u_i), lookup_user_index(u2idx, u_j)
                    if u_i is not None and u_j is not None:
                        simedges[mapped_interest].append((u_i, u_j))

    for interest, edges in list(simedges.items()):
        unique_edges = list(set(edges))
        if len(unique_edges) < min_edges:
            simedges.pop(interest)
        else:
            simedges[interest] = unique_edges

    if missing_topics:
        logger.info("Topics missing from mapping: %s", len(missing_topics))
    logger.info("simedges: %s", len(simedges))
    logger.info("Keys in simedges:")
    for key in simedges.keys():
        logger.info(key)
    return simedges


def validate_shuffle_ratio(shuffle_ratio: float):
    if shuffle_ratio < 0 or shuffle_ratio > 1:
        raise ValueError(f"--shuffle-ratio must be in [0, 1], got {shuffle_ratio}")


def corrupt_one_edge(
    edge: tuple,
    existing_edges: set,
    forbidden_edges: set,
    user_size: int,
    corrupt_endpoint: str,
    rng: random.Random,
):
    if user_size <= 1:
        raise ValueError("Cannot corrupt edges when user_size <= 1.")

    u, v = edge
    endpoint = rng.choice(["source", "target"]) if corrupt_endpoint == "either" else corrupt_endpoint

    for _ in range(max(100, user_size * 2)):
        if endpoint == "source":
            candidate = (rng.randrange(user_size), v)
        else:
            candidate = (u, rng.randrange(user_size))
        if (
            candidate != edge
            and candidate not in existing_edges
            and candidate not in forbidden_edges
            and candidate[0] != candidate[1]
        ):
            return candidate

    endpoints = ["source", "target"] if corrupt_endpoint == "either" else [endpoint]
    for endpoint in endpoints:
        if endpoint == "source":
            for new_u in range(user_size):
                candidate = (new_u, v)
                if (
                    candidate != edge
                    and candidate not in existing_edges
                    and candidate not in forbidden_edges
                    and candidate[0] != candidate[1]
                ):
                    return candidate
        else:
            for new_v in range(user_size):
                candidate = (u, new_v)
                if (
                    candidate != edge
                    and candidate not in existing_edges
                    and candidate not in forbidden_edges
                    and candidate[0] != candidate[1]
                ):
                    return candidate

    raise ValueError(f"Cannot find a valid corrupted replacement for edge {edge}.")


def maybe_shuffle_edges(
    simedges: dict,
    shuffle_ratio: float,
    user_size: int,
    corrupt_endpoint: str,
    random_seed: Optional[int],
    forbidden_edges: Optional[set] = None,
):
    validate_shuffle_ratio(shuffle_ratio)
    if shuffle_ratio <= 0:
        return simedges
    if corrupt_endpoint not in {"source", "target", "either"}:
        raise ValueError("--corrupt-endpoint must be one of: source, target, either")

    rng = random.Random(random_seed)
    forbidden_edges = forbidden_edges or set()

    for interest, edges in simedges.items():
        if not edges:
            continue
        edges = list(set(edges))
        num_shuffle = int(len(edges) * shuffle_ratio + 0.5)
        if num_shuffle == 0:
            simedges[interest] = edges
            continue
        shuffle_indices = rng.sample(range(len(edges)), num_shuffle)
        edge_set = set(edges)
        for corrupt_i, edge_idx in enumerate(shuffle_indices, start=1):
            old_edge = edges[edge_idx]
            edge_set.remove(old_edge)
            new_edge = corrupt_one_edge(
                old_edge,
                existing_edges=edge_set,
                forbidden_edges=forbidden_edges,
                user_size=user_size,
                corrupt_endpoint=corrupt_endpoint,
                rng=rng,
            )
            edges[edge_idx] = new_edge
            edge_set.add(new_edge)
            if corrupt_i % 100000 == 0:
                logger.info("Corrupting semantic edges for %s: %s/%s", interest, corrupt_i, num_shuffle)
        simedges[interest] = edges
        logger.info(
            "Corrupted semantic edges for %s: %s/%s (ratio=%.4f, endpoint=%s)",
            interest,
            num_shuffle,
            len(edges),
            shuffle_ratio,
            corrupt_endpoint,
        )
    return simedges


def convert_to_graph(
    simedges: dict,
    user_size: int,
    original_edges: Optional[list] = None,
    shuffle_ratio: float = DEFAULT_SHUFFLE_RATIO,
    corrupt_endpoint: str = "either",
    random_seed: Optional[int] = None,
):
    forbidden_edges = {tuple(edge) for edge in original_edges} if original_edges else set()
    simedges = maybe_shuffle_edges(
        simedges,
        shuffle_ratio=shuffle_ratio,
        user_size=user_size,
        corrupt_endpoint=corrupt_endpoint,
        random_seed=random_seed,
        forbidden_edges=forbidden_edges,
    )

    if original_edges:
        for interest, edges in simedges.items():
            simedges[interest] = list(set(edges + original_edges))

    for interest, edges in simedges.items():
        edges += [(u, u) for u in range(user_size)]
        simedges[interest] = list(set(edges))

    graph_d = {}
    for interest, edges in simedges.items():
        edge_index = torch.LongTensor(list(zip(*edges)))
        edge_weight = torch.FloatTensor([1] * edge_index.size(1))
        graph_d[interest] = Data(edge_index=edge_index, edge_weight=edge_weight)

    logger.info("Keys in graph_d:")
    for key in graph_d.keys():
        logger.info(key)
    return graph_d


def infer_user_size(idx2u, u2idx=None):
    if u2idx is not None:
        return len(u2idx)
    if isinstance(idx2u, dict):
        return len(idx2u)
    return len(list(idx2u))


def build_parser():
    default_dataset_dir = os.path.join(DEFAULT_DATA_ROOT, DATASET_NAME)
    parser = argparse.ArgumentParser(description="Build Weibo topic diffusion graphs from LLM topic assignments.")
    parser.add_argument("--param", type=int, required=True, help="Confidence threshold and output version suffix.")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Time window length in days.")
    parser.add_argument("--min-edges", type=int, default=10, help="Drop mapped interests with fewer edges.")
    parser.add_argument("--shuffle-ratio", type=float, default=DEFAULT_SHUFFLE_RATIO, help="Exact semantic edge corruption ratio.")
    parser.add_argument(
        "--corrupt-endpoint",
        choices=["source", "target", "either"],
        default="either",
        help="Which endpoint to alter when corrupting semantic edges.",
    )
    parser.add_argument("--random-seed", type=int, default=2023, help="Random seed for semantic edge corruption.")
    parser.add_argument("--dataset-dir", type=str, default=default_dataset_dir)
    parser.add_argument("--result-file", type=str, default=os.path.join(PPX_SHARE_ROOT, "result_weibo.jsonl"))
    parser.add_argument("--mapping-file", type=str, default=None)
    parser.add_argument("--wiki-file", type=str, default=DEFAULT_WIKI_FILE)
    parser.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--force-remap", action="store_true", help="Rebuild gt_ft_mapping_ppx.data even if it exists.")
    parser.add_argument("--u2idx-file", type=str, default=None)
    parser.add_argument("--idx2u-file", type=str, default=None)
    parser.add_argument("--edges-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--user-size", type=int, default=None)
    parser.add_argument("--no-map-users", action="store_true", help="Use result ids directly instead of mapping through u2idx.")
    parser.add_argument("--no-original-edges", action="store_true", help="Do not merge social-network edges into topic graphs.")
    return parser


def main():
    args = build_parser().parse_args()
    topic_dir = os.path.join(args.dataset_dir, "topic_llm_ppx")
    mapping_file = args.mapping_file or os.path.join(topic_dir, "gt_ft_mapping_ppx.data")
    u2idx_file = args.u2idx_file or os.path.join(args.dataset_dir, "u2idx.data")
    idx2u_file = args.idx2u_file or os.path.join(args.dataset_dir, "idx2u.data")
    edges_file = args.edges_file or os.path.join(args.dataset_dir, "edges.data")
    output_file = args.output_file or os.path.join(
        topic_dir,
        f"{GRAPH_PREFIX}{args.param}_new_topic_diffusion_graph_full_windowsize{args.window_days}.data",
    )

    logger.info("Result file: %s", args.result_file)
    logger.info("Mapping file: %s", mapping_file)
    logger.info("Output file: %s", output_file)

    results = parse_llm_results(args.result_file, args.param)
    topic_mapping = load_or_build_topic_mapping(
        mapping_file,
        result_file=args.result_file,
        wiki_file=args.wiki_file,
        sbert_model=args.sbert_model,
        force_remap=args.force_remap,
    )
    u2idx = None if args.no_map_users else load_pickle(u2idx_file)
    idx2u = load_pickle(idx2u_file)
    user_size = args.user_size or infer_user_size(idx2u, u2idx)
    logger.info("max_user_size: %s", user_size)

    original_edges = None if args.no_original_edges else load_pickle(edges_file)
    time_distance = 3600 * 24 * args.window_days
    interest_cascades = generate_interest_cascades_nomapping(results)
    simedges = generate_interest_simedges(
        interest_cascades,
        topic_mapping,
        time_distance=time_distance,
        min_edges=args.min_edges,
        u2idx=u2idx,
    )
    graph_d = convert_to_graph(
        simedges,
        user_size=user_size,
        original_edges=original_edges,
        shuffle_ratio=args.shuffle_ratio,
        corrupt_endpoint=args.corrupt_endpoint,
        random_seed=args.random_seed,
    )
    for _, graph in graph_d.items():
        logger.info(graph.edge_index.size())
    save_pickle(graph_d, output_file)
    print("The file has saved.")


if __name__ == "__main__":
    main()
