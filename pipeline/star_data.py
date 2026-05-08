"""Data preparation utilities for star-graph training.

Reads existing artifacts:
- data/star_graphs_gnn.pkl
- data/star_graph_edge_splits.json

Builds model-ready sample dicts with keys:
- h0, h_known, y_known, h_target, y
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class StarDataConfig:
    gnn_payload_path: Path = Path("data/star_graphs_gnn.pkl")
    edge_splits_path: Path = Path("data/star_graph_edge_splits.json")
    neighbor_cache_path: Path = Path("data/star_graph_neighbor_cache.pkl")
    num_classes: int = 188


@dataclass
class TrainEpochSampler:
    """Train-only sampler that re-masks one positive edge per star occurrence each epoch.

    Validation and test continue using fixed held-out edges from the saved split file.
    """

    graph_lookup: dict[int, dict]
    neighbor_cache: dict[int, dict[int, int]]
    positive_edges_by_graph: dict[int, list[dict]]
    positive_graph_sequence: list[int]
    fixed_negative_samples: list[dict]
    max_samples: int = 0

    def sample(self, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        samples = []
        for graph_id in self.positive_graph_sequence:
            graph = self.graph_lookup.get(int(graph_id))
            edge_candidates = self.positive_edges_by_graph.get(int(graph_id), [])
            if graph is None or not edge_candidates:
                continue
            edge_item = edge_candidates[int(rng.integers(0, len(edge_candidates)))]
            center_rel_map = self.neighbor_cache.get(int(graph_id))
            samples.append(build_sample_from_edge(graph, edge_item, center_rel_map=center_rel_map))

        if self.fixed_negative_samples:
            samples.extend(self.fixed_negative_samples)

        if self.max_samples and self.max_samples > 0:
            return samples[: self.max_samples]
        return samples

    @property
    def num_samples(self) -> int:
        total = len(self.positive_graph_sequence) + len(self.fixed_negative_samples)
        if self.max_samples and self.max_samples > 0:
            return min(total, self.max_samples)
        return total


def load_star_payload(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_edge_splits(path: Path) -> dict:
    return json.loads(path.read_text())


def _graph_cache_path(cfg: StarDataConfig, split: str | None = None) -> Path:
    if not split:
        return cfg.neighbor_cache_path
    suffix = cfg.neighbor_cache_path.suffix
    stem = cfg.neighbor_cache_path.stem
    return cfg.neighbor_cache_path.with_name(f"{stem}_{split}{suffix}")


def _get_graphs_for_split(gnn_payload: dict, split: str | None = None) -> list[dict]:
    graphs_by_split = gnn_payload.get("graphs_by_split")
    if isinstance(graphs_by_split, dict) and split in graphs_by_split:
        return graphs_by_split[split]
    if split == "train" and "graphs_train" in gnn_payload:
        return gnn_payload.get("graphs_train", [])
    return gnn_payload.get("graphs_all", [])


def build_graph_lookup(gnn_payload: dict, split: str | None = None) -> dict[int, dict]:
    graphs = _get_graphs_for_split(gnn_payload, split=split)
    return {int(g["graph_id"]): g for g in graphs}


def _build_center_neighbor_relation_map(graph: dict) -> dict[int, int]:
    src = np.asarray(graph["src"], dtype=np.int64)
    dst = np.asarray(graph["dst"], dtype=np.int64)
    rel = np.asarray(graph["rel_type"], dtype=np.int64)

    mapping = {}
    for s, d, r in zip(src, dst, rel):
        if int(s) == 0 and int(d) != 0:
            mapping[int(d)] = int(r)
    return mapping


def build_or_load_neighbor_cache(
    cfg: StarDataConfig,
    graph_lookup: dict[int, dict],
    split: str | None = None,
    force_recompute: bool = False,
) -> dict[int, dict[int, int]]:
    cache_path = _graph_cache_path(cfg, split)
    if cache_path.exists() and not force_recompute:
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                return cached
        except Exception:
            pass

    neighbor_cache = {}
    for graph_id, graph in graph_lookup.items():
        neighbor_cache[int(graph_id)] = _build_center_neighbor_relation_map(graph)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(neighbor_cache, f)
    return neighbor_cache


def _safe_target_index(dst: int, num_nodes: int) -> int:
    if 0 <= int(dst) < int(num_nodes):
        return int(dst)
    # fallback for malformed samples
    return min(max(1, int(dst)), int(num_nodes) - 1)


def build_sample_from_edge(
    graph: dict,
    edge_item: dict,
    center_rel_map: dict[int, int] | None = None,
) -> dict[str, np.ndarray | int]:
    node_feat = np.asarray(graph["node_feat"], dtype=np.float32)
    num_nodes = int(node_feat.shape[0])
    target_node = _safe_target_index(int(edge_item["dst"]), num_nodes)

    # Center and target states
    h0 = node_feat[0]
    h_target = node_feat[target_node]

    # Known neighbors: all non-center nodes except the target node.
    if center_rel_map is None:
        center_rel_map = _build_center_neighbor_relation_map(graph)
    known_nodes = [n for n in range(1, num_nodes) if n != target_node and n in center_rel_map]

    if not known_nodes:
        known_nodes = [target_node]

    h_known = node_feat[np.asarray(known_nodes, dtype=np.int64)]
    y_known = np.asarray([center_rel_map.get(n, 0) for n in known_nodes], dtype=np.int64)

    return {
        "h0": h0,
        "h0_id": graph.get("graph_id", -1), # NEW: Capture the original drug ID
        "h_known": h_known,
        "y_known": y_known,
        "h_target": h_target,
        "y": np.int64(edge_item["relation"]),
    }


def _iter_edge_items(split_bucket: dict):
    n = len(split_bucket["graph_id"])
    for i in range(n):
        yield {
            "graph_id": int(split_bucket["graph_id"][i]),
            "src": int(split_bucket["src"][i]),
            "dst": int(split_bucket["dst"][i]),
            "relation": int(split_bucket["relation"][i]),
            "label": int(split_bucket["label"][i]),
        }


def build_split_samples(
    cfg: StarDataConfig,
    split: str = "train",
    include_negative: bool = False,
    use_neighbor_cache: bool = True,
    force_recompute_cache: bool = False,
) -> list[dict]:
    gnn_payload = load_star_payload(cfg.gnn_payload_path)
    edge_splits = load_edge_splits(cfg.edge_splits_path)
    graph_lookup = build_graph_lookup(gnn_payload, split=split)
    neighbor_cache = (
        build_or_load_neighbor_cache(
            cfg,
            graph_lookup,
            split=split,
            force_recompute=force_recompute_cache,
        )
        if use_neighbor_cache
        else {}
    )

    split_obj = edge_splits[split]
    buckets = ["positive", "negative"] if include_negative else ["positive"]

    samples = []
    for bucket in buckets:
        for edge_item in _iter_edge_items(split_obj[bucket]):
            graph = graph_lookup.get(edge_item["graph_id"])
            if graph is None:
                continue
            center_rel_map = neighbor_cache.get(int(edge_item["graph_id"]))
            samples.append(build_sample_from_edge(graph, edge_item, center_rel_map=center_rel_map))

    return samples


def build_train_epoch_sampler(
    cfg: StarDataConfig,
    include_negative: bool = False,
    use_neighbor_cache: bool = True,
    force_recompute_cache: bool = False,
    max_samples: int = 0,
) -> TrainEpochSampler:
    """Build a sampler that randomly re-selects one positive held-out edge per train sample slot.

    The train graph distribution stays fixed, but the held-out positive edge is re-sampled
    within each train graph on every epoch. Negative samples, if requested, remain fixed.
    """
    gnn_payload = load_star_payload(cfg.gnn_payload_path)
    edge_splits = load_edge_splits(cfg.edge_splits_path)
    graph_lookup = build_graph_lookup(gnn_payload, split="train")
    neighbor_cache = (
        build_or_load_neighbor_cache(
            cfg,
            graph_lookup,
            split="train",
            force_recompute=force_recompute_cache,
        )
        if use_neighbor_cache
        else {}
    )

    train_split = edge_splits["train"]
    positive_edges_by_graph: dict[int, list[dict]] = defaultdict(list)
    positive_graph_sequence: list[int] = []
    for edge_item in _iter_edge_items(train_split["positive"]):
        graph_id = int(edge_item["graph_id"])
        positive_graph_sequence.append(graph_id)
        positive_edges_by_graph[graph_id].append(edge_item)

    fixed_negative_samples: list[dict] = []
    if include_negative:
        for edge_item in _iter_edge_items(train_split["negative"]):
            graph = graph_lookup.get(int(edge_item["graph_id"]))
            if graph is None:
                continue
            center_rel_map = neighbor_cache.get(int(edge_item["graph_id"]))
            fixed_negative_samples.append(
                build_sample_from_edge(graph, edge_item, center_rel_map=center_rel_map)
            )

    return TrainEpochSampler(
        graph_lookup=graph_lookup,
        neighbor_cache=neighbor_cache,
        positive_edges_by_graph=dict(positive_edges_by_graph),
        positive_graph_sequence=positive_graph_sequence,
        fixed_negative_samples=fixed_negative_samples,
        max_samples=max_samples,
    )


def collate_samples(samples: list[dict], device: str = "cpu") -> dict[str, Tensor]:
    if not samples:
        raise ValueError("No samples to collate")

    max_n = max(s["h_known"].shape[0] for s in samples)
    d_model = samples[0]["h0"].shape[0]

    bsz = len(samples)
    h0 = np.zeros((bsz, d_model), dtype=np.float32)
    h0_ids = []
    h_target = np.zeros((bsz, d_model), dtype=np.float32)
    h_known = np.zeros((bsz, max_n, d_model), dtype=np.float32)
    y_known = np.zeros((bsz, max_n), dtype=np.int64)
    known_mask = np.zeros((bsz, max_n), dtype=bool)
    y = np.zeros((bsz,), dtype=np.int64)

    for i, sample in enumerate(samples):
        n_i = sample["h_known"].shape[0]
        h0[i] = sample["h0"]
        h_target[i] = sample["h_target"]
        h_known[i, :n_i] = sample["h_known"]
        y_known[i, :n_i] = sample["y_known"]
        known_mask[i, :n_i] = True
        y[i] = sample["y"]
        h0_ids.append(sample.get("h0_id", -1))

    return {
        "h0": torch.from_numpy(h0).to(device),
        "h0_ids": torch.tensor(h0_ids, device=device),
        "h_known": torch.from_numpy(h_known).to(device),
        "y_known": torch.from_numpy(y_known).to(device),
        "known_mask": torch.from_numpy(known_mask).to(device),
        "h_target": torch.from_numpy(h_target).to(device),
        "y": torch.from_numpy(y).to(device),
    }
