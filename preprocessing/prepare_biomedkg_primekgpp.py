import argparse
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare PrimeKG++-style KG artifacts from BioMedKG data under kg/. "
            "This script does not use SLGNN KG files."
        )
    )
    parser.add_argument(
        "--kg-data-dir",
        type=Path,
        default=Path("kg/data"),
        help="BioMedKG data folder containing embed/ and gcl_embed/.",
    )
    parser.add_argument(
        "--triplets-source",
        type=str,
        default="primekg",
        choices=["primekg"],
        help=(
            "Triplet source mode. Only 'primekg' is supported to ensure "
            "BioMedKG-only preprocessing."
        ),
    )
    parser.add_argument(
        "--triplets-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit PrimeKG-style triplets path under kg/. "
            "If omitted, the script builds triplets from --primekg-csv-path."
        ),
    )
    parser.add_argument(
        "--primekg-csv-path",
        type=Path,
        default=Path("kg/data/primekg/kg.csv"),
        help="Path for PrimeKG CSV used to build triplets when --triplets-source=primekg.",
    )
    parser.add_argument(
        "--primekg-download-url",
        type=str,
        default="https://dataverse.harvard.edu/api/access/datafile/6180620",
        help="URL used to download PrimeKG CSV when missing.",
    )
    parser.add_argument(
        "--primekg-node-types",
        type=str,
        default="gene/protein,drug,disease",
        help="Comma-separated node types filter for PrimeKG triplets.",
    )
    parser.add_argument(
        "--subgraph-hops",
        type=int,
        default=0,
        help=(
            "If > 0, build an induced k-hop subgraph around seed nodes. "
            "When enabled with --triplets-source=primekg, triplets are loaded from full PrimeKG "
            "(node-type filter is ignored) before subgraph extraction."
        ),
    )
    parser.add_argument(
        "--subgraph-min-seed-matches",
        type=int,
        default=1,
        help=(
            "Minimum number of seed nodes that must match graph nodes when --subgraph-hops > 0. "
            "Set to 0 to allow empty seed matches without raising an error."
        ),
    )
    parser.add_argument(
        "--target-drug-map",
        type=Path,
        default=Path("tahoe_to_primekg_map.csv"),
        help=(
            "CSV used to validate that all target drugs are present in PrimeKG nodes. "
            "Columns supported: x_name (preferred) or drug."
        ),
    )
    parser.add_argument(
        "--disable-full-primekg-fallback",
        action="store_true",
        help="Disable automatic fallback to full PrimeKG when target drug coverage is incomplete.",
    )
    parser.add_argument(
        "--node-map-path",
        type=Path,
        default=None,
        help=(
            "Optional CSV/TSV mapping numeric node IDs to names (columns: node_id,node_name). "
            "Useful when triplets are integer IDs but embeddings are keyed by names."
        ),
    )
    parser.add_argument(
        "--embedding-source",
        type=str,
        default="grace_redaf",
        help=(
            "Embedding source key. Examples: grace_none, grace_attention, ggd_none, "
            "dgi_none, primekg_modality_lm. Default is grace_redaf (recommended)."
        ),
    )
    parser.add_argument(
        "--l2-normalize-node-embeddings",
        action="store_true",
        help="Apply L2 normalization to final node embeddings before export (disabled by default).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Target embedding dimension. If omitted, infer from loaded embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic initialization of missing node embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/primekgpp_slgnn"),
        help="Output directory for prepared artifacts.",
    )
    return parser.parse_args()


def _assert_biomedkg_kg_path(path: Path):
    resolved = path.resolve()
    resolved_str = str(resolved)

    if "reference/SLGNN" in resolved_str or "/SLGNN/" in resolved_str:
        raise ValueError(
            f"Invalid path {path}: SLGNN KG inputs are not allowed for this preprocessing."
        )

    if "star_graph" in resolved.name:
        raise ValueError(
            f"Invalid path {path}: local star-graph artifacts are not valid PrimeKG++ triplets."
        )

    kg_root = Path("kg").resolve()
    try:
        resolved.relative_to(kg_root)
    except ValueError as exc:
        raise ValueError(
            f"Invalid path {path}: expected an input under {kg_root} (BioMedKG kg folder)."
        ) from exc


def _parse_node_types(raw_node_types: str):
    return [item.strip() for item in raw_node_types.split(",") if item.strip()]


def load_target_drug_names(map_path: Path):
    if not map_path.exists():
        return []

    df = pd.read_csv(map_path)
    lower_to_col = {c.lower(): c for c in df.columns}

    target_col = lower_to_col.get("x_name") or lower_to_col.get("drug")
    if target_col is None:
        return []

    values = [str(v).strip() for v in df[target_col].dropna().tolist() if str(v).strip()]
    return sorted(set(values))


def find_missing_target_drugs(node_names, target_drugs):
    if not target_drugs:
        return []

    node_name_norm = {str(name).strip().casefold() for name in node_names}
    missing = [drug for drug in target_drugs if drug.strip().casefold() not in node_name_norm]
    return sorted(set(missing))


def ensure_primekg_csv(csv_path: Path, download_url: str):
    if csv_path.exists():
        _assert_biomedkg_kg_path(csv_path)
        return csv_path

    _assert_biomedkg_kg_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tdc.resource import PrimeKG

        print("PrimeKG CSV not found; downloading via TDC PrimeKG API...")
        primekg_obj = PrimeKG(path=str(csv_path.parent))
        primekg_obj.df.to_csv(csv_path, index=False)
        return csv_path
    except Exception:
        pass

    print(f"PrimeKG CSV not found at {csv_path}. Downloading from {download_url}...")
    try:
        urllib.request.urlretrieve(download_url, str(csv_path))
    except Exception as exc:
        raise RuntimeError(
            "Failed to obtain PrimeKG CSV. Use one of:\n"
            "1) hf download tienda02/BioMedKG --repo-type dataset --local-dir ./kg/data\n"
            "2) provide --primekg-csv-path to a local kg.csv under kg/\n"
            f"Original error: {exc}"
        ) from exc
    return csv_path


def load_primekg_triplets(primekg_csv_path: Path, node_types: list[str]):
    df = pd.read_csv(primekg_csv_path, low_memory=False)

    required_cols = {"x_name", "x_type", "relation", "y_name", "y_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"PrimeKG CSV missing required columns: {sorted(missing)}")

    if node_types:
        df = df[df["x_type"].isin(node_types) & df["y_type"].isin(node_types)]

    triplets = []
    for _, row in df[["x_name", "relation", "y_name"]].dropna().iterrows():
        triplets.append(
            _normalize_triplet_fields(row["x_name"], row["relation"], row["y_name"])
        )

    if len(triplets) == 0:
        raise ValueError("No triplets were generated from PrimeKG with the selected node types.")

    return triplets


def resolve_and_load_triplets(args):
    if args.triplets_path is not None:
        _assert_biomedkg_kg_path(args.triplets_path)
        if not args.triplets_path.exists():
            raise FileNotFoundError(f"Triplets file not found: {args.triplets_path}")
        triplets = load_triplets(args.triplets_path)
        return triplets, args.triplets_path, "primekg_custom"

    primekg_csv = ensure_primekg_csv(args.primekg_csv_path, args.primekg_download_url)
    node_types = _parse_node_types(args.primekg_node_types)
    if args.subgraph_hops > 0:
        triplets = load_primekg_triplets(primekg_csv, node_types=[])
        return triplets, primekg_csv, "primekg_full_for_subgraph"

    triplets = load_primekg_triplets(primekg_csv, node_types=node_types)
    return triplets, primekg_csv, "primekg"


def _collect_graph_nodes(triplets):
    nodes = set()
    for h, _, t in triplets:
        nodes.add(str(h))
        nodes.add(str(t))
    return nodes


def _resolve_seed_nodes(graph_nodes, candidate_seeds):
    if not candidate_seeds:
        return []

    norm_to_node = {node.strip().casefold(): node for node in graph_nodes}
    matched = []
    seen = set()
    for seed in candidate_seeds:
        norm = str(seed).strip().casefold()
        node = norm_to_node.get(norm)
        if node is not None and node not in seen:
            matched.append(node)
            seen.add(node)
    return matched


def _build_undirected_adjacency(triplets):
    adjacency = {}
    for h, _, t in triplets:
        h = str(h)
        t = str(t)
        adjacency.setdefault(h, set()).add(t)
        adjacency.setdefault(t, set()).add(h)
    return adjacency


def _collect_k_hop_nodes(adjacency, seed_nodes, hops):
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(adjacency.get(node, set()))
        next_frontier -= visited
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier

    return visited


def build_khop_subgraph_triplets(triplets, seed_candidates, hops: int, min_seed_matches: int = 1):
    if hops <= 0:
        return triplets, {
            "enabled": False,
            "hops": int(hops),
            "seed_candidates": int(len(seed_candidates or [])),
            "seed_matches": 0,
            "kept_nodes": 0,
            "kept_triplets": int(len(triplets)),
        }

    graph_nodes = _collect_graph_nodes(triplets)
    seed_nodes = _resolve_seed_nodes(graph_nodes, seed_candidates)
    if len(seed_nodes) < int(min_seed_matches):
        raise ValueError(
            "Subgraph extraction failed: matched seed nodes "
            f"{len(seed_nodes)} < required minimum {int(min_seed_matches)}."
        )

    adjacency = _build_undirected_adjacency(triplets)
    keep_nodes = _collect_k_hop_nodes(adjacency, seed_nodes, hops=hops)

    filtered_triplets = [
        (h, r, t)
        for (h, r, t) in triplets
        if str(h) in keep_nodes and str(t) in keep_nodes
    ]

    if len(filtered_triplets) == 0:
        raise ValueError(
            "Subgraph extraction produced zero triplets. "
            "Check seed mapping and --subgraph-hops value."
        )

    stats = {
        "enabled": True,
        "hops": int(hops),
        "seed_candidates": int(len(seed_candidates or [])),
        "seed_matches": int(len(seed_nodes)),
        "kept_nodes": int(len(keep_nodes)),
        "kept_triplets": int(len(filtered_triplets)),
    }
    return filtered_triplets, stats


def load_node_id_to_name(node_map_path: Path | None):
    if node_map_path is None:
        return {}
    if not node_map_path.exists():
        raise FileNotFoundError(f"Node map not found: {node_map_path}")

    sep = "\t" if node_map_path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(node_map_path, sep=sep)
    lower_to_col = {c.lower(): c for c in df.columns}

    id_col = lower_to_col.get("node_id") or lower_to_col.get("id")
    name_col = lower_to_col.get("node_name") or lower_to_col.get("name")

    if id_col is None or name_col is None:
        raise ValueError(
            "Node map must contain columns node_id and node_name (or id/name)."
        )

    return {
        str(row[id_col]): str(row[name_col])
        for _, row in df[[id_col, name_col]].dropna().iterrows()
    }


def _normalize_triplet_fields(head, relation, tail):
    return str(head).strip(), str(relation).strip(), str(tail).strip()


def _load_triplets_json(path: Path):
    raw = json.loads(path.read_text())

    if isinstance(raw, dict):
        for key in ("triplets", "edges", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("JSON triplets must be a list or contain list under triplets/edges/data.")

    out = []
    for item in raw:
        if isinstance(item, dict):
            h = item.get("head", item.get("h", item.get("src")))
            r = item.get("relation", item.get("r", item.get("edge_type")))
            t = item.get("tail", item.get("t", item.get("dst")))
            if h is None or r is None or t is None:
                continue

            if (
                "plate" in item
                and "cell_line" in item
                and "src" in item
                and "dst" in item
                and "head" not in item
                and "tail" not in item
            ):
                plate = str(item["plate"])
                cell_line = str(item["cell_line"])
                h = f"{plate}|{cell_line}|node_{h}"
                t = f"{plate}|{cell_line}|node_{t}"

            out.append(_normalize_triplet_fields(h, r, t))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append(_normalize_triplet_fields(item[0], item[1], item[2]))

    return out


def _load_triplets_delimited(path: Path):
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".txt", ".tsv"} else ","

    df = pd.read_csv(path, sep=sep)
    if len(df.columns) < 3:
        df = pd.read_csv(path, sep=sep, names=["h", "r", "t"], header=None)

    cols_lower = {c.lower(): c for c in df.columns}

    head_col = (
        cols_lower.get("h")
        or cols_lower.get("head")
        or cols_lower.get("x_name")
        or cols_lower.get("source")
        or cols_lower.get("src")
        or df.columns[0]
    )
    rel_col = (
        cols_lower.get("r")
        or cols_lower.get("relation")
        or cols_lower.get("edge_type")
        or df.columns[1]
    )
    tail_col = (
        cols_lower.get("t")
        or cols_lower.get("tail")
        or cols_lower.get("y_name")
        or cols_lower.get("target")
        or cols_lower.get("dst")
        or df.columns[2]
    )

    out = []
    for _, row in df[[head_col, rel_col, tail_col]].dropna().iterrows():
        out.append(_normalize_triplet_fields(row[head_col], row[rel_col], row[tail_col]))
    return out


def load_triplets(path: Path):
    if path.suffix.lower() == ".json":
        triplets = _load_triplets_json(path)
    else:
        triplets = _load_triplets_delimited(path)

    if len(triplets) == 0:
        raise ValueError(f"No valid triplets were parsed from {path}")
    return triplets


def load_embedding_mapping(kg_data_dir: Path, embedding_source: str):
    gcl_path = kg_data_dir / "gcl_embed" / f"{embedding_source}.pickle"
    if gcl_path.exists():
        with gcl_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Unexpected GCL pickle format in {gcl_path}")
        return obj, gcl_path

    if embedding_source == "primekg_modality_lm":
        lm_path = kg_data_dir / "embed" / "primekg_modality_lm.pickle"
    else:
        lm_path = kg_data_dir / "embed" / f"{embedding_source}.pickle"

    if lm_path.exists():
        with lm_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Unexpected embedding pickle format in {lm_path}")
        return obj, lm_path

    available = []
    for folder in [kg_data_dir / "gcl_embed", kg_data_dir / "embed"]:
        if folder.exists():
            available.extend(sorted([p.name for p in folder.glob("*.pickle")]))

    raise FileNotFoundError(
        "Embedding source not found. "
        f"Requested: {embedding_source}. Available pickles: {available}"
    )


def _to_1d_embedding(value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0).astype(np.float32)
    raise ValueError(f"Unsupported embedding shape: {arr.shape}")


def infer_embedding_dim(embedding_map, requested_dim: int | None):
    if requested_dim is not None:
        return requested_dim

    for value in embedding_map.values():
        emb = _to_1d_embedding(value)
        if emb.size > 0:
            return int(emb.shape[0])

    raise ValueError("Could not infer embedding dim from embedding source; set --embedding-dim.")


def normalize_embedding_map(embedding_map, embedding_dim: int):
    normalized = {}
    for key, value in embedding_map.items():
        try:
            emb = _to_1d_embedding(value)
        except ValueError:
            continue

        if emb.shape[0] == embedding_dim:
            normalized[str(key)] = emb
        elif emb.shape[0] > embedding_dim:
            normalized[str(key)] = emb[:embedding_dim]
        else:
            padded = np.zeros(embedding_dim, dtype=np.float32)
            padded[: emb.shape[0]] = emb
            normalized[str(key)] = padded
    return normalized


def build_tables(triplets, node_id_to_name):
    resolved_triplets = []
    for h, r, t in triplets:
        h_name = node_id_to_name.get(h, h)
        t_name = node_id_to_name.get(t, t)
        resolved_triplets.append((h_name, r, t_name))

    node_names = sorted({h for h, _, _ in resolved_triplets} | {t for _, _, t in resolved_triplets})
    relations = sorted({r for _, r, _ in resolved_triplets})

    node_to_id = {name: idx for idx, name in enumerate(node_names)}
    rel_to_id = {name: idx for idx, name in enumerate(relations)}

    edge_rows = [
        (node_to_id[h], rel_to_id[r], node_to_id[t], h, r, t)
        for (h, r, t) in resolved_triplets
    ]

    edge_df = pd.DataFrame(
        edge_rows,
        columns=["head_id", "relation_id", "tail_id", "head_name", "relation_name", "tail_name"],
    )
    node_df = pd.DataFrame(
        [(idx, name) for name, idx in node_to_id.items()],
        columns=["node_id", "node_name"],
    ).sort_values("node_id")
    rel_df = pd.DataFrame(
        [(idx, name) for name, idx in rel_to_id.items()],
        columns=["relation_id", "relation_name"],
    ).sort_values("relation_id")

    return node_df, rel_df, edge_df


def build_node_embeddings(
    node_df,
    normalized_embedding_map,
    embedding_dim: int,
    seed: int,
    l2_normalize: bool = False,
):
    rng = np.random.default_rng(seed)

    n = len(node_df)
    matrix = np.zeros((n, embedding_dim), dtype=np.float32)
    from_pretrained = np.zeros(n, dtype=np.int8)

    for row in node_df.itertuples(index=False):
        node_id = int(row.node_id)
        node_name = str(row.node_name)

        emb = normalized_embedding_map.get(node_name)
        if emb is None:
            emb = rng.normal(0.0, 0.02, size=(embedding_dim,)).astype(np.float32)
        else:
            from_pretrained[node_id] = 1

        if l2_normalize:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

        matrix[node_id] = emb

    return matrix, from_pretrained


def export_outputs(
    output_dir: Path,
    node_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_embeddings: np.ndarray,
    from_pretrained: np.ndarray,
    meta: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    slgnn_triples = edge_df[["head_id", "relation_id", "tail_id"]]
    slgnn_path = output_dir / "kg2id.txt"
    slgnn_triples.to_csv(slgnn_path, sep="\t", header=False, index=False)

    np.save(output_dir / "node_embeddings.npy", node_embeddings)
    node_df = node_df.copy()
    node_df["from_pretrained"] = from_pretrained
    node_df.to_csv(output_dir / "node_index.csv", index=False)
    rel_df.to_csv(output_dir / "relation_index.csv", index=False)
    edge_df.to_csv(output_dir / "edges_detailed.csv", index=False)

    with (output_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def export_kg_bundle(
    output_dir: Path,
    node_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_embeddings: np.ndarray,
):
    entity2id = node_df[["node_name", "node_id"]].rename(
        columns={"node_name": "entity_name", "node_id": "entity_id"}
    )
    relation2id = rel_df[["relation_name", "relation_id"]]
    triplets_named = edge_df[["head_name", "relation_name", "tail_name"]]

    entity2id.to_csv(output_dir / "entity2id.txt", sep="\t", header=False, index=False)
    relation2id.to_csv(output_dir / "relation2id.txt", sep="\t", header=False, index=False)
    triplets_named.to_csv(output_dir / "triplets_named.tsv", sep="\t", header=False, index=False)

    embedding_dict = {
        row.node_name: node_embeddings[int(row.node_id)]
        for row in node_df.itertuples(index=False)
    }
    with (output_dir / "embedding_dict.pickle").open("wb") as f:
        pickle.dump(embedding_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()

    triplets, triplets_origin_path, triplets_origin_kind = resolve_and_load_triplets(args)

    node_id_to_name = load_node_id_to_name(args.node_map_path)
    target_drugs = load_target_drug_names(args.target_drug_map)

    subgraph_stats = {
        "enabled": False,
        "hops": int(args.subgraph_hops),
        "seed_candidates": int(len(target_drugs)),
        "seed_matches": 0,
        "kept_nodes": 0,
        "kept_triplets": int(len(triplets)),
    }
    if args.subgraph_hops > 0:
        triplets, subgraph_stats = build_khop_subgraph_triplets(
            triplets=triplets,
            seed_candidates=target_drugs,
            hops=args.subgraph_hops,
            min_seed_matches=args.subgraph_min_seed_matches,
        )
        triplets_origin_kind = f"{triplets_origin_kind}_khop"

    raw_embedding_map, embedding_path = load_embedding_mapping(
        kg_data_dir=args.kg_data_dir,
        embedding_source=args.embedding_source,
    )

    embedding_dim = infer_embedding_dim(raw_embedding_map, args.embedding_dim)
    normalized_embedding_map = normalize_embedding_map(raw_embedding_map, embedding_dim)

    node_df, rel_df, edge_df = build_tables(triplets=triplets, node_id_to_name=node_id_to_name)

    missing_target_drugs = find_missing_target_drugs(node_df["node_name"].tolist(), target_drugs)

    if missing_target_drugs:
        print(
            "Target-drug coverage check: "
            f"{len(missing_target_drugs)} missing out of {len(target_drugs)} mapped drugs "
            f"using source {triplets_origin_kind}."
        )

        can_fallback = (
            triplets_origin_kind == "primekg"
            and bool(_parse_node_types(args.primekg_node_types))
            and not args.disable_full_primekg_fallback
        )

        if can_fallback:
            print("Retrying with full PrimeKG (no node-type filter) to recover missing drugs...")
            full_triplets = load_primekg_triplets(triplets_origin_path, node_types=[])
            node_df, rel_df, edge_df = build_tables(
                triplets=full_triplets,
                node_id_to_name=node_id_to_name,
            )
            missing_target_drugs = find_missing_target_drugs(
                node_df["node_name"].tolist(),
                target_drugs,
            )
            triplets_origin_kind = "primekg_full"
    node_embeddings, from_pretrained = build_node_embeddings(
        node_df=node_df,
        normalized_embedding_map=normalized_embedding_map,
        embedding_dim=embedding_dim,
        seed=args.seed,
        l2_normalize=args.l2_normalize_node_embeddings,
    )

    coverage_ratio = float(from_pretrained.sum()) / float(len(from_pretrained)) if len(from_pretrained) else 0.0

    meta = {
        "triplets_path": str(triplets_origin_path),
        "embedding_source": args.embedding_source,
        "embedding_path": str(embedding_path),
        "embedding_dim": int(embedding_dim),
        "num_nodes": int(node_df.shape[0]),
        "num_relations": int(rel_df.shape[0]),
        "num_edges": int(edge_df.shape[0]),
        "num_pretrained_nodes": int(from_pretrained.sum()),
        "pretrained_coverage_ratio": coverage_ratio,
        "seed": int(args.seed),
        "target_drug_map_path": str(args.target_drug_map),
        "target_drug_count": int(len(target_drugs)),
        "missing_target_drug_count": int(len(missing_target_drugs)),
        "missing_target_drugs_preview": missing_target_drugs[:50],
        "l2_normalize_node_embeddings": bool(args.l2_normalize_node_embeddings),
        "subgraph_enabled": bool(subgraph_stats["enabled"]),
        "subgraph_hops": int(subgraph_stats["hops"]),
        "subgraph_seed_candidates": int(subgraph_stats["seed_candidates"]),
        "subgraph_seed_matches": int(subgraph_stats["seed_matches"]),
        "subgraph_kept_nodes": int(subgraph_stats["kept_nodes"]),
        "subgraph_kept_triplets": int(subgraph_stats["kept_triplets"]),
    }

    export_outputs(
        output_dir=args.output_dir,
        node_df=node_df,
        rel_df=rel_df,
        edge_df=edge_df,
        node_embeddings=node_embeddings,
        from_pretrained=from_pretrained,
        meta=meta,
    )
    export_kg_bundle(
        output_dir=args.output_dir,
        node_df=node_df,
        rel_df=rel_df,
        edge_df=edge_df,
        node_embeddings=node_embeddings,
    )

    meta["triplets_origin_kind"] = triplets_origin_kind
    meta["triplets_origin_path"] = str(triplets_origin_path)
    with (args.output_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("PrimeKG++-style preparation complete")
    print(f"Triplets source      : {triplets_origin_path} ({triplets_origin_kind})")
    print(f"Embeddings source    : {embedding_path}")
    print(f"Output dir           : {args.output_dir}")
    print(f"Nodes / Relations    : {meta['num_nodes']} / {meta['num_relations']}")
    print(f"Edges                : {meta['num_edges']}")
    if meta["subgraph_enabled"]:
        print(
            "Subgraph             : "
            f"{meta['subgraph_hops']}-hop, seeds matched {meta['subgraph_seed_matches']}/"
            f"{meta['subgraph_seed_candidates']}, kept triplets {meta['subgraph_kept_triplets']}"
        )
    print(
        "Target-drug coverage : "
        f"{meta['target_drug_count'] - meta['missing_target_drug_count']}/"
        f"{meta['target_drug_count']}"
    )
    print(
        "Pretrained coverage  : "
        f"{meta['num_pretrained_nodes']}/{meta['num_nodes']} "
        f"({meta['pretrained_coverage_ratio']:.2%})"
    )
    print("Saved files          : kg2id.txt, entity2id.txt, relation2id.txt, triplets_named.tsv,")
    print("                       node_embeddings.npy, embedding_dict.pickle, node_index.csv,")
    print("                       relation_index.csv, edges_detailed.csv, meta.json")


if __name__ == "__main__":
    main()
