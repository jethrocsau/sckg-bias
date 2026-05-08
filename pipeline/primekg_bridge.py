"""PrimeKG bridge utilities.

Purpose:
- load PrimeKG++ artifacts,
- map 188 local classes to PrimeKG node IDs,
- assemble H_KG class-signature tensor for Stage 2 routing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass
class PrimeKGBridgeConfig:
    primekg_dir: Path = Path("data/primekgpp_grace_redaf")
    map_csv_path: Path = Path("tahoe_to_primekg_map.csv")
    num_classes: int = 188

    def __post_init__(self):
        """Automatically resolve paths based on the provided primekg_dir."""
        self.entity2id_path = self.primekg_dir / "entity2id.txt"
        self.node_embeddings_path = self.primekg_dir / "node_embeddings.npy"
        self.kg2id_path = self.primekg_dir / "kg2id.txt"
        self.relation2id_path = self.primekg_dir / "relation2id.txt"
        self.hkg_cache_path = self.primekg_dir / "hkg_class_3hop.npy"
        self.hkg_multirel_cache_path = self.primekg_dir / "hkg_class_multirel_3hop.npz"


def _load_local_to_primekg_map(cfg: PrimeKGBridgeConfig) -> dict[str, str]:
    map_df = pd.read_csv(cfg.map_csv_path)
    col_lookup = {c.lower(): c for c in map_df.columns}
    local_col = col_lookup.get("drug")
    primekg_col = col_lookup.get("x_name") or local_col
    if local_col is None or primekg_col is None:
        raise ValueError("Mapping CSV must contain 'drug' and/or 'x_name' columns")

    local_to_primekg = {}
    for row in map_df.itertuples(index=False):
        local_name = str(getattr(row, local_col)).strip().casefold()
        primekg_name = str(getattr(row, primekg_col)).strip().casefold()
        local_to_primekg[local_name] = primekg_name
    return local_to_primekg


def _resolve_class_node_ids(
    cfg: PrimeKGBridgeConfig,
    class_names: list[str],
) -> tuple[dict[str, str], dict[str, int], list[int | None]]:
    local_to_primekg = _load_local_to_primekg_map(cfg)
    entity_name_to_id = load_entity_name_to_id(cfg.entity2id_path)

    class_node_ids: list[int | None] = []
    for class_name in class_names:
        local_key = str(class_name).strip().casefold()
        primekg_key = local_to_primekg.get(local_key, local_key)
        class_node_ids.append(entity_name_to_id.get(primekg_key))

    return local_to_primekg, entity_name_to_id, class_node_ids


def _resolve_multirel_cache_path(
    cfg: PrimeKGBridgeConfig,
    hops: int,
    relation_limit: int | None,
) -> Path:
    cache_path = cfg.hkg_multirel_cache_path
    suffix = cache_path.suffix or ".npz"
    stem = cache_path.stem
    tag = f"{stem}.h{hops}"
    if relation_limit is not None:
        tag = f"{tag}.top{relation_limit}"
    return cache_path.with_name(f"{tag}{suffix}")


def load_multirel_relation_ids(
    cfg: PrimeKGBridgeConfig,
    hops: int = 3,
    relation_limit: int | None = None,
) -> list[int]:
    """Load retained relation ids from the cached multirelation H_KG tensor."""
    cache_path = _resolve_multirel_cache_path(cfg, hops=hops, relation_limit=relation_limit)
    if not cache_path.exists():
        return []
    with np.load(cache_path, allow_pickle=False) as cached:
        relation_ids = cached.get("relation_ids")
        if relation_ids is None:
            return []
        return [int(relation_id) for relation_id in relation_ids.tolist()]


def load_entity_name_to_id(entity2id_path: Path) -> dict[str, int]:
    """Parse entity2id with names containing tabs safely (split from right)."""
    mapping: dict[str, int] = {}
    with entity2id_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, node_id = line.rsplit("\t", 1)
            mapping[name.strip().casefold()] = int(node_id)
    return mapping


def load_relation_id_to_name(relation2id_path: Path) -> dict[int, str]:
    """Parse relation2id into id -> relation name."""
    mapping: dict[int, str] = {}
    with relation2id_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, relation_id = line.rsplit("\t", 1)
            mapping[int(relation_id)] = name.strip()
    return mapping


def load_drug_class_order(meta_path: Path = Path("data/star_graphs_meta.json")) -> list[str]:
    """Return class order by index from edge_drug_to_index."""
    meta = json.loads(meta_path.read_text())
    edge_drug_to_index = meta["edge_drug_to_index"]

    class_names = [None] * len(edge_drug_to_index)
    for drug, idx in edge_drug_to_index.items():
        class_names[int(idx)] = str(drug)
    return class_names


def build_h_kg_class_tensor(cfg: PrimeKGBridgeConfig, class_names: list[str]) -> Tensor:
    """Build class-aligned H_KG tensor [C, d_kg] from name mapping.

    Unmapped classes currently get zero vectors as deterministic fallback.
    """
    local_to_primekg, entity_name_to_id, _ = _resolve_class_node_ids(cfg, class_names)
    node_embeddings = np.load(cfg.node_embeddings_path)
    d_kg = int(node_embeddings.shape[1])

    h_kg = np.zeros((len(class_names), d_kg), dtype=np.float32)
    for class_idx, class_name in enumerate(class_names):
        local_key = str(class_name).strip().casefold()
        primekg_key = local_to_primekg.get(local_key, local_key)
        node_id = entity_name_to_id.get(primekg_key)
        if node_id is not None and 0 <= node_id < node_embeddings.shape[0]:
            h_kg[class_idx] = node_embeddings[node_id]

    return torch.from_numpy(h_kg)


def _load_adjacency(
    kg2id_path: Path,
    num_nodes: int,
) -> list[dict[int, set[int]]]:
    adj: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(num_nodes)]
    with kg2id_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            head_id = int(parts[0])
            relation_id = int(parts[1])
            tail_id = int(parts[2])
            if 0 <= head_id < num_nodes and 0 <= tail_id < num_nodes:
                adj[head_id][relation_id].add(tail_id)
                adj[tail_id][relation_id].add(head_id)
    return adj


def _k_hop_relation_nodes(
    adj: list[dict[int, set[int]]],
    root: int,
    hops: int,
) -> dict[int, list[tuple[int, int]]]:
    """Traverses neighborhood and tracks the hop distance for each neighbor."""
    visited = {root: 0}
    frontier = {root}
    # relation_id -> list of (neighbor_id, hop_distance)
    relation_nodes: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for h in range(1, hops + 1):
        nxt = set()
        for node in frontier:
            for relation_id, neighbors in adj[node].items():
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = h
                        relation_nodes[int(relation_id)].append((int(neighbor), h))
                        nxt.add(int(neighbor))
        if not nxt:
            break
        frontier = nxt

    return relation_nodes


def precompute_h_kg_multihop(
    cfg: PrimeKGBridgeConfig,
    class_names: list[str],
    hops: int = 3,
    decay_factor: float = 0.1,
    relation_limit: int | None = None,
    force_recompute: bool = False,
) -> Tensor:
    """Precompute H_KG using multi-hop aggregation with distance-based decay.
    Reserves Index 0 for the drug's own node embedding (Identity).
    """
    if hops <= 0:
        raise ValueError("hops must be >= 1")

    cache_path = _resolve_multirel_cache_path(cfg, hops=hops, relation_limit=relation_limit)

    if cache_path.exists() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as cached:
            h_kg_multi = cached["h_kg_multi"]
        if h_kg_multi.shape[0] == len(class_names):
            return torch.from_numpy(h_kg_multi.astype(np.float32, copy=False))

    # Setup PrimeKG artifacts
    _, _, class_node_ids = _resolve_class_node_ids(cfg, class_names)
    node_embeddings = np.load(cfg.node_embeddings_path).astype(np.float32, copy=False)
    num_nodes = int(node_embeddings.shape[0])
    adj = _load_adjacency(cfg.kg2id_path, num_nodes)

    class_relation_nodes: list[dict[int, list[tuple[int, int]]]] = []
    relation_support: dict[int, int] = defaultdict(int)

    # 1. Neighborhood Traversal (Context)
    for node_id in class_node_ids:
        if node_id is None or not (0 <= node_id < num_nodes):
            class_relation_nodes.append({})
            continue

        rel_nodes = _k_hop_relation_nodes(adj, node_id, hops=hops)
        class_relation_nodes.append(rel_nodes)
        for relation_id, neighbors in rel_nodes.items():
            relation_support[int(relation_id)] += len(neighbors)

    # 2. Relation Selection
    retained_relation_ids = [
        relation_id
        for relation_id, _ in sorted(relation_support.items(), key=lambda item: (-item[1], item[0]))
    ]
    if relation_limit is not None and relation_limit > 0:
        retained_relation_ids = retained_relation_ids[:relation_limit]

    # 3. Weighted Aggregation with Identity Offset
    # Note: relation_to_index starts at 1 to reserve 0 for Self
    relation_to_index = {relation_id: idx + 1 for idx, relation_id in enumerate(retained_relation_ids)}
    h_kg_multi = np.zeros(
        (len(class_names), len(retained_relation_ids) + 1, int(node_embeddings.shape[1])),
        dtype=np.float32,
    )

    for class_idx, node_id in enumerate(class_node_ids):
        # self embedding
        if node_id is not None and 0 <= node_id < num_nodes:
            h_kg_multi[class_idx, 0] = node_embeddings[node_id]

        # neighbour nodes
        rel_nodes = class_relation_nodes[class_idx]
        for relation_id, neighbors in rel_nodes.items():
            relation_idx = relation_to_index.get(int(relation_id))
            if relation_idx is None or not neighbors:
                continue
            
            neighbor_ids = np.array([n[0] for n in neighbors], dtype=np.int64)
            hop_counts = np.array([n[1] for n in neighbors], dtype=np.float32)
            weights = (decay_factor ** hop_counts).reshape(-1, 1) # [N, 1]

            embeddings = node_embeddings[neighbor_ids] 
            weighted_emb = (embeddings * weights).sum(axis=0) / weights.sum()
            
            h_kg_multi[class_idx, relation_idx] = weighted_emb

    # 4. Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        h_kg_multi=h_kg_multi,
        relation_ids=np.asarray(retained_relation_ids, dtype=np.int64),
    )
    return torch.from_numpy(h_kg_multi)