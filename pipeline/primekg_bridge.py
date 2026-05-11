"""PrimeKG bridge utilities.

Purpose:
- load PrimeKG++ artifacts,
- map 188 local classes to PrimeKG node IDs,
- assemble H_KG class-signature tensor for Stage 2 routing.
"""

from __future__ import annotations

from collections import defaultdict
from collections import deque
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
        self.hkg_path_cache_path = self.primekg_dir / "hkg_class_paths_3hop.npz"


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
    tag = f"{stem}.rootrelv2.h{hops}"
    if relation_limit is not None:
        tag = f"{tag}.top{relation_limit}"
    return cache_path.with_name(f"{tag}{suffix}")


def _resolve_path_cache_path(
    cfg: PrimeKGBridgeConfig,
    hops: int,
    max_paths: int,
    relation_limit: int | None,
) -> Path:
    cache_path = cfg.hkg_path_cache_path
    suffix = cache_path.suffix or ".npz"
    stem = cache_path.stem
    tag = f"{stem}.h{hops}.top{max_paths}"
    if relation_limit is not None:
        tag = f"{tag}.rootreltop{relation_limit}"
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


def load_multirel_relation_mask(
    cfg: PrimeKGBridgeConfig,
    hops: int = 3,
    relation_limit: int | None = None,
) -> Tensor | None:
    """Load retained relation mask from the cached multirelation H_KG tensor."""
    cache_path = _resolve_multirel_cache_path(cfg, hops=hops, relation_limit=relation_limit)
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        relation_mask = cached.get("relation_mask")
        if relation_mask is None:
            return None
        return torch.from_numpy(relation_mask.astype(bool, copy=False))


def load_multirel_relation_context(
    cfg: PrimeKGBridgeConfig,
    hops: int = 3,
    relation_limit: int | None = None,
) -> dict[str, Tensor] | None:
    """Load cached ragged subtree-member tensors for hierarchical relation pooling."""
    cache_path = _resolve_multirel_cache_path(cfg, hops=hops, relation_limit=relation_limit)
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as cached:
        required = [
            "root_embeddings",
            "member_offsets",
            "member_embeddings_flat",
            "member_weights_flat",
            "relation_mask",
        ]
        if any(cached.get(key) is None for key in required):
            return None

        context = {
            "root_embeddings": torch.from_numpy(cached["root_embeddings"].astype(np.float32, copy=False)),
            "member_offsets": torch.from_numpy(cached["member_offsets"].astype(np.int64, copy=False)),
            "member_embeddings_flat": torch.from_numpy(cached["member_embeddings_flat"].astype(np.float32, copy=False)),
            "member_weights_flat": torch.from_numpy(cached["member_weights_flat"].astype(np.float32, copy=False)),
            "rel_mask": torch.from_numpy(cached["relation_mask"].astype(bool, copy=False)),
        }

        h_kg_multi = cached.get("h_kg_multi")
        if h_kg_multi is not None:
            context["rel_embeddings"] = torch.from_numpy(h_kg_multi.astype(np.float32, copy=False))

        member_hops_flat = cached.get("member_hops_flat")
        if member_hops_flat is not None:
            context["member_hops_flat"] = torch.from_numpy(member_hops_flat.astype(np.int64, copy=False))

        member_node_ids_flat = cached.get("member_node_ids_flat")
        if member_node_ids_flat is not None:
            context["member_node_ids_flat"] = torch.from_numpy(member_node_ids_flat.astype(np.int64, copy=False))

        relation_ids = cached.get("relation_ids")
        if relation_ids is not None:
            context["relation_ids"] = torch.from_numpy(relation_ids.astype(np.int64, copy=False))

        return context


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


def load_entity_id_to_name(entity2id_path: Path) -> dict[int, str]:
    """Parse entity2id into id -> entity name."""
    mapping: dict[int, str] = {}
    with entity2id_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, node_id = line.rsplit("\t", 1)
            mapping[int(node_id)] = name.strip()
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


def _dfs_root_relation_nodes(
    adj: list[dict[int, set[int]]],
    root: int,
    hops: int,
) -> dict[int, list[tuple[int, int]]]:
    """Traverse from `root` by first-hop relation and aggregate reached nodes by DFS.

    Each returned bucket corresponds to an immediate relation off the root node, and
    contains `(node_id, hop_distance)` pairs for all nodes reached when starting the
    traversal through that root relation.
    """
    relation_nodes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if hops <= 0:
        return relation_nodes

    for root_relation, first_neighbors in adj[root].items():
        visited_depth: dict[int, int] = {}
        stack: list[tuple[int, int, frozenset[int]]] = []
        for neighbor_id in first_neighbors:
            stack.append((int(neighbor_id), 1, frozenset({int(root), int(neighbor_id)})))

        while stack:
            node_id, depth, path_nodes = stack.pop()
            best_depth = visited_depth.get(node_id)
            if best_depth is not None and best_depth <= depth:
                continue

            visited_depth[node_id] = depth
            relation_nodes[int(root_relation)].append((int(node_id), int(depth)))

            if depth >= hops:
                continue

            for next_neighbors in adj[node_id].values():
                for next_node in next_neighbors:
                    next_node = int(next_node)
                    if next_node in path_nodes:
                        continue
                    stack.append((next_node, depth + 1, path_nodes | {next_node}))

    return relation_nodes


def _k_hop_paths(
    adj: list[dict[int, set[int]]],
    root: int,
    hops: int,
    max_paths: int,
    retained_root_relations: set[int] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Collect simple shortest-first paths up to `hops`."""
    if hops <= 0 or max_paths <= 0:
        return []

    queue = deque([(root, [], [root])])
    paths: list[tuple[list[int], list[int]]] = []
    seen_signatures: set[tuple[int, ...]] = set()

    while queue and len(paths) < max_paths:
        node_id, relation_seq, node_seq = queue.popleft()
        depth = len(relation_seq)

        if depth > 0:
            signature = tuple(node_seq)
            root_relation = int(relation_seq[0]) if relation_seq else None
            is_retained = retained_root_relations is None or root_relation in retained_root_relations
            if is_retained and signature not in seen_signatures:
                seen_signatures.add(signature)
                paths.append((list(node_seq), list(relation_seq)))
                if len(paths) >= max_paths:
                    break

        if depth == hops:
            continue

        for relation_id in sorted(adj[node_id]):
            for neighbor_id in sorted(adj[node_id][relation_id]):
                if neighbor_id in node_seq:
                    continue
                queue.append(
                    (
                        int(neighbor_id),
                        [*relation_seq, int(relation_id)],
                        [*node_seq, int(neighbor_id)],
                    )
                )

    return paths


def precompute_h_kg_multihop(
    cfg: PrimeKGBridgeConfig,
    class_names: list[str],
    hops: int = 3,
    decay_factor: float = 0.1,
    relation_limit: int | None = None,
    force_recompute: bool = False,
) -> Tensor:
    """Precompute per-drug KG relation hints using DFS rooted at immediate relations.

    Each retained row corresponds to one immediate relation off the indexed drug in
    PrimeKG. The row embedding is built as:

        concat(root_drug_embedding, decay_averaged_subtree_embedding)

    so the downstream relation selector sees an explicit per-relation representation.
    """
    if hops <= 0:
        raise ValueError("hops must be >= 1")

    cache_path = _resolve_multirel_cache_path(cfg, hops=hops, relation_limit=relation_limit)

    if cache_path.exists() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as cached:
            h_kg_multi = cached["h_kg_multi"]
            has_hier_members = all(
                cached.get(key) is not None
                for key in ["root_embeddings", "member_offsets", "member_embeddings_flat", "member_weights_flat"]
            )
        if h_kg_multi.shape[0] == len(class_names) and has_hier_members:
            return torch.from_numpy(h_kg_multi.astype(np.float32, copy=False))

    # Setup PrimeKG artifacts
    _, _, class_node_ids = _resolve_class_node_ids(cfg, class_names)
    node_embeddings = np.load(cfg.node_embeddings_path).astype(np.float32, copy=False)
    num_nodes = int(node_embeddings.shape[0])
    base_d_kg = int(node_embeddings.shape[1])
    adj = _load_adjacency(cfg.kg2id_path, num_nodes)

    class_relation_nodes: list[dict[int, list[tuple[int, int]]]] = []
    relation_support: dict[int, int] = defaultdict(int)

    # 1. Rooted DFS traversal grouped by the immediate relation from each drug node.
    for node_id in class_node_ids:
        if node_id is None or not (0 <= node_id < num_nodes):
            class_relation_nodes.append({})
            continue

        rel_nodes = _dfs_root_relation_nodes(adj, int(node_id), hops=hops)
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

    # 3. Weighted Aggregation by retained root relation.
    relation_to_index = {relation_id: idx for idx, relation_id in enumerate(retained_relation_ids)}
    num_classes = len(class_names)
    num_relations = len(retained_relation_ids)
    h_kg_multi = np.zeros((num_classes, num_relations, base_d_kg * 2), dtype=np.float32)
    relation_mask = np.zeros((len(class_names), len(retained_relation_ids)), dtype=bool)
    root_embeddings = np.zeros((num_classes, base_d_kg), dtype=np.float32)
    member_offsets = np.zeros((num_classes, num_relations, 2), dtype=np.int64)
    member_embeddings_flat: list[np.ndarray] = []
    member_weights_flat: list[float] = []
    member_hops_flat: list[int] = []
    member_node_ids_flat: list[int] = []

    for class_idx, node_id in enumerate(class_node_ids):
        if node_id is None or not (0 <= node_id < num_nodes):
            continue

        root_emb = node_embeddings[int(node_id)]
        root_embeddings[class_idx] = root_emb
        rel_nodes = class_relation_nodes[class_idx]
        for relation_id, neighbors in rel_nodes.items():
            relation_idx = relation_to_index.get(int(relation_id))
            if relation_idx is None or not neighbors:
                continue

            start_idx = len(member_embeddings_flat)
            neighbor_ids = np.array([n[0] for n in neighbors], dtype=np.int64)
            hop_counts = np.array([n[1] for n in neighbors], dtype=np.float32)
            weights = decay_factor ** hop_counts

            embeddings = node_embeddings[neighbor_ids]
            member_embeddings_flat.extend(embeddings)
            member_weights_flat.extend(weights.tolist())
            member_hops_flat.extend(hop_counts.astype(np.int64).tolist())
            member_node_ids_flat.extend(neighbor_ids.tolist())

            weighted_emb = (embeddings * weights.reshape(-1, 1)).sum(axis=0) / weights.sum()

            h_kg_multi[class_idx, relation_idx] = np.concatenate([root_emb, weighted_emb], axis=0)
            relation_mask[class_idx, relation_idx] = True
            member_offsets[class_idx, relation_idx, 0] = start_idx
            member_offsets[class_idx, relation_idx, 1] = len(member_embeddings_flat)

    if member_embeddings_flat:
        member_embeddings_arr = np.asarray(member_embeddings_flat, dtype=np.float32)
        member_weights_arr = np.asarray(member_weights_flat, dtype=np.float32)
        member_hops_arr = np.asarray(member_hops_flat, dtype=np.int64)
        member_node_ids_arr = np.asarray(member_node_ids_flat, dtype=np.int64)
    else:
        member_embeddings_arr = np.zeros((0, base_d_kg), dtype=np.float32)
        member_weights_arr = np.zeros((0,), dtype=np.float32)
        member_hops_arr = np.zeros((0,), dtype=np.int64)
        member_node_ids_arr = np.zeros((0,), dtype=np.int64)

    # 4. Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        h_kg_multi=h_kg_multi,
        root_embeddings=root_embeddings,
        member_offsets=member_offsets,
        member_embeddings_flat=member_embeddings_arr,
        member_weights_flat=member_weights_arr,
        member_hops_flat=member_hops_arr,
        member_node_ids_flat=member_node_ids_arr,
        relation_mask=relation_mask,
        relation_ids=np.asarray(retained_relation_ids, dtype=np.int64),
    )
    return torch.from_numpy(h_kg_multi)


def precompute_h_kg_paths(
    cfg: PrimeKGBridgeConfig,
    class_names: list[str],
    hops: int = 3,
    decay_factor: float = 0.5,
    max_paths: int = 32,
    relation_limit: int | None = None,
    force_recompute: bool = False,
) -> dict[str, Tensor]:
    """Precompute per-class path tokens for path-level KG attention."""
    if hops <= 0:
        raise ValueError("hops must be >= 1")
    if max_paths <= 0:
        raise ValueError("max_paths must be >= 1")

    cache_path = _resolve_path_cache_path(cfg, hops=hops, max_paths=max_paths, relation_limit=relation_limit)
    if cache_path.exists() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as cached:
            path_embeddings = cached["path_embeddings"]
        if path_embeddings.shape[0] == len(class_names):
            with np.load(cache_path, allow_pickle=False) as cached:
                return {
                    "path_embeddings": torch.from_numpy(cached["path_embeddings"].astype(np.float32, copy=False)),
                    "path_mask": torch.from_numpy(cached["path_mask"].astype(bool, copy=False)),
                    "endpoint_ids": torch.from_numpy(cached["endpoint_ids"].astype(np.int64, copy=False)),
                    "path_hops": torch.from_numpy(cached["path_hops"].astype(np.int64, copy=False)),
                    "relation_seqs": torch.from_numpy(cached["relation_seqs"].astype(np.int64, copy=False)),
                    "node_seqs": torch.from_numpy(cached["node_seqs"].astype(np.int64, copy=False)),
                }

    _, _, class_node_ids = _resolve_class_node_ids(cfg, class_names)
    node_embeddings = np.load(cfg.node_embeddings_path).astype(np.float32, copy=False)
    num_nodes = int(node_embeddings.shape[0])
    adj = _load_adjacency(cfg.kg2id_path, num_nodes)
    d_kg = int(node_embeddings.shape[1])

    class_relation_nodes: list[dict[int, list[tuple[int, int]]]] = []
    relation_support: dict[int, int] = defaultdict(int)
    for node_id in class_node_ids:
        if node_id is None or not (0 <= node_id < num_nodes):
            class_relation_nodes.append({})
            continue

        rel_nodes = _dfs_root_relation_nodes(adj, int(node_id), hops=hops)
        class_relation_nodes.append(rel_nodes)
        for relation_id, neighbors in rel_nodes.items():
            relation_support[int(relation_id)] += len(neighbors)

    retained_relation_ids = [
        relation_id
        for relation_id, _ in sorted(relation_support.items(), key=lambda item: (-item[1], item[0]))
    ]
    if relation_limit is not None and relation_limit > 0:
        retained_relation_ids = retained_relation_ids[:relation_limit]
    retained_relation_set = set(retained_relation_ids)

    num_classes = len(class_names)
    path_embeddings = np.zeros((num_classes, max_paths, d_kg), dtype=np.float32)
    path_mask = np.zeros((num_classes, max_paths), dtype=bool)
    endpoint_ids = np.full((num_classes, max_paths), -1, dtype=np.int64)
    path_hops = np.zeros((num_classes, max_paths), dtype=np.int64)
    relation_seqs = np.full((num_classes, max_paths, hops), -1, dtype=np.int64)
    node_seqs = np.full((num_classes, max_paths, hops + 1), -1, dtype=np.int64)

    for class_idx, node_id in enumerate(class_node_ids):
        if node_id is None or not (0 <= node_id < num_nodes):
            continue

        retained_paths = _k_hop_paths(
            adj,
            int(node_id),
            hops=hops,
            max_paths=max_paths,
            retained_root_relations=retained_relation_set if retained_relation_set else None,
        )

        for path_idx, (node_seq, relation_seq) in enumerate(retained_paths):
            node_arr = np.asarray(node_seq, dtype=np.int64)
            depth_arr = np.arange(len(node_seq), dtype=np.float32)
            weights = (decay_factor ** depth_arr).reshape(-1, 1)
            emb = node_embeddings[node_arr]
            path_embeddings[class_idx, path_idx] = (emb * weights).sum(axis=0) / weights.sum()
            path_mask[class_idx, path_idx] = True
            endpoint_ids[class_idx, path_idx] = int(node_seq[-1])
            path_hops[class_idx, path_idx] = len(relation_seq)
            relation_seqs[class_idx, path_idx, : len(relation_seq)] = np.asarray(relation_seq, dtype=np.int64)
            node_seqs[class_idx, path_idx, : len(node_seq)] = node_arr

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        path_embeddings=path_embeddings,
        path_mask=path_mask,
        endpoint_ids=endpoint_ids,
        path_hops=path_hops,
        relation_seqs=relation_seqs,
        node_seqs=node_seqs,
        relation_ids=np.asarray(retained_relation_ids, dtype=np.int64),
    )

    return {
        "path_embeddings": torch.from_numpy(path_embeddings),
        "path_mask": torch.from_numpy(path_mask),
        "endpoint_ids": torch.from_numpy(endpoint_ids),
        "path_hops": torch.from_numpy(path_hops),
        "relation_seqs": torch.from_numpy(relation_seqs),
        "node_seqs": torch.from_numpy(node_seqs),
    }