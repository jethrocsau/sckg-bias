import argparse
import ast
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_tuple_key(raw_key: str):
    try:
        parsed = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, tuple) or len(parsed) < 3:
        return None

    cell_line = str(parsed[0])
    drug = str(parsed[1])
    plate = str(parsed[2])
    return cell_line, drug, plate


def mean_embedding(array_like):
    arr = np.asarray(array_like, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0)
    raise ValueError(f"Unsupported embedding ndim: {arr.ndim}")


def load_condition_means(embedding_paths: list[Path]):
    condition_means = {}

    for embeddings_path in embedding_paths:
        try:
            data_obj = np.load(embeddings_path, allow_pickle=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to read embeddings NPZ: {embeddings_path}. "
                "The file may be incomplete or corrupted."
            ) from exc

        with data_obj as data:
            for raw_key in data.files:
                key = parse_tuple_key(raw_key)
                if key is None:
                    continue

                emb = mean_embedding(data[raw_key])
                if key in condition_means:
                    condition_means[key] = (condition_means[key] + emb) / 2.0
                else:
                    condition_means[key] = emb

    return condition_means


def resolve_input_paths(user_input: Path):
    if any(token in str(user_input) for token in ["*", "?", "["]):
        glob_paths = sorted(Path(p) for p in glob.glob(str(user_input)))
        glob_paths = [p for p in glob_paths if p.suffix.lower() == ".npz"]
        if glob_paths:
            return glob_paths

    if user_input.exists():
        if user_input.is_dir():
            shard_paths = sorted(user_input.glob("*.npz"))
            if shard_paths:
                return shard_paths
            raise FileNotFoundError(f"No .npz shards found in directory: {user_input}")
        return [user_input]

    shard_candidate = sorted(user_input.parent.glob(f"{user_input.stem}_*.npz"))
    if shard_candidate:
        return shard_candidate

    candidates = [
        sorted(Path("data").glob("tahoe_embeddings_parquet*.npz")),
        [Path("data/tahoe_embeddings_parquet.npz")],
    ]
    for candidate_group in candidates:
        valid = [path for path in candidate_group if path.exists()]
        if valid:
            return valid

    raise FileNotFoundError(
        "Input embeddings not found. Looked for: "
        f"{user_input}, matching shard patterns, and default candidates in data/."
    )


def build_grouped_conditions(condition_means):
    grouped = defaultdict(dict)
    for (cell_line, drug, plate), embedding in condition_means.items():
        grouped[(plate, cell_line)][drug] = embedding
    return grouped


def load_drug_mapping(map_path: Path, center_drugs):
    if not map_path.exists():
        raise FileNotFoundError(f"Target drug mapping file not found: {map_path}")

    df = pd.read_csv(map_path)
    if "drug" not in df.columns:
        raise ValueError(f"Expected 'drug' column in {map_path}")

    drugs = sorted({str(d).strip() for d in df["drug"].dropna().tolist() if str(d).strip()})
    drugs = [d for d in drugs if d not in center_drugs]
    if not drugs:
        raise ValueError("No non-center drugs found in target drug mapping CSV.")

    return {drug: idx for idx, drug in enumerate(drugs)}


def one_hot(index: int, length: int):
    vec = np.zeros(length, dtype=np.float32)
    vec[index] = 1.0
    return vec


def build_star_graphs(grouped_conditions, center_drugs, drug_to_idx):
    candidate_graphs = []

    for (plate, cell_line), drug_to_emb in grouped_conditions.items():
        available_centers = [drug_to_emb[d] for d in center_drugs if d in drug_to_emb]
        if not available_centers:
            continue

        center_embedding = np.mean(np.vstack(available_centers), axis=0).astype(np.float32)

        perturbed = {
            d: emb
            for d, emb in drug_to_emb.items()
            if d not in center_drugs and d in drug_to_idx
        }
        if not perturbed:
            continue

        candidate_graphs.append((plate, cell_line, center_embedding, perturbed))

    graph_records = []
    positive_edges = []
    for graph_id, (plate, cell_line, center_embedding, perturbed) in enumerate(candidate_graphs):
        perturbed_items = sorted(perturbed.items(), key=lambda item: item[0])

        node_embeddings = [center_embedding]
        node_drugs = ["DMSO_CENTER"]
        edge_src, edge_dst = [], []
        edge_attrs = []
        edge_drugs = []

        for node_index, (drug, emb) in enumerate(perturbed_items, start=1):
            node_embeddings.append(np.asarray(emb, dtype=np.float32))
            node_drugs.append(drug)

            attr = one_hot(drug_to_idx[drug], len(drug_to_idx))
            edge_src.extend([0, node_index])
            edge_dst.extend([node_index, 0])
            edge_attrs.extend([attr, attr.copy()])
            edge_drugs.extend([drug, drug])

            relation_idx = drug_to_idx[drug]
            positive_edges.append(
                {
                    "graph_id": graph_id,
                    "plate": plate,
                    "cell_line": cell_line,
                    "src": 0,
                    "relation": relation_idx,
                    "dst": node_index,
                }
            )

        graph_records.append(
            {
                "plate": plate,
                "cell_line": cell_line,
                "node_embeddings": np.vstack(node_embeddings).astype(np.float32),
                "node_drugs": np.array(node_drugs, dtype=object),
                "edge_index": np.vstack(
                    [np.asarray(edge_src, dtype=np.int64), np.asarray(edge_dst, dtype=np.int64)]
                ),
                "edge_attr": np.vstack(edge_attrs).astype(np.float32),
                "edge_drugs": np.array(edge_drugs, dtype=object),
                "center_node": 0,
                "graph_id": graph_id,
            }
        )

    return graph_records, positive_edges


def _negative_sample_for_edge(pos_edge, graph_record, positive_set):
    src = int(pos_edge["src"])
    rel = int(pos_edge["relation"])
    n_nodes = int(graph_record["node_embeddings"].shape[0])
    if n_nodes <= 1:
        return None

    start = (src + rel + 1) % n_nodes
    for offset in range(n_nodes):
        candidate_dst = int((start + offset) % n_nodes)
        if candidate_dst == src:
            continue
        if (src, rel, candidate_dst) in positive_set:
            continue
        return {
            "graph_id": int(pos_edge["graph_id"]),
            "plate": pos_edge["plate"],
            "cell_line": pos_edge["cell_line"],
            "src": src,
            "relation": rel,
            "dst": candidate_dst,
        }
    return None


def _serialize_edges(edge_records, relation_dim, label):
    n = len(edge_records)
    graph_id = np.zeros(n, dtype=np.int64)
    src = np.zeros(n, dtype=np.int64)
    dst = np.zeros(n, dtype=np.int64)
    rel = np.zeros(n, dtype=np.int64)
    y = np.full(n, int(label), dtype=np.int64)
    edge_feat = np.zeros((n, relation_dim), dtype=np.float32)
    plate = np.empty(n, dtype=object)
    cell_line = np.empty(n, dtype=object)

    for i, e in enumerate(edge_records):
        graph_id[i] = int(e["graph_id"])
        src[i] = int(e["src"])
        dst[i] = int(e["dst"])
        rel[i] = int(e["relation"])
        edge_feat[i, rel[i]] = 1.0
        plate[i] = e["plate"]
        cell_line[i] = e["cell_line"]

    return {
        "graph_id": graph_id,
        "src": src,
        "dst": dst,
        "relation": rel,
        "edge_feat": edge_feat,
        "label": y,
        "plate": plate,
        "cell_line": cell_line,
    }


def _partition_ids(all_ids, rng, train_ratio, val_ratio):
    ids = np.asarray(list(all_ids), dtype=np.int64)
    rng.shuffle(ids)

    train_end = int(len(ids) * train_ratio)
    val_end = train_end + int(len(ids) * val_ratio)
    return {
        "train": set(ids[:train_end].tolist()),
        "val": set(ids[train_end:val_end].tolist()),
        "test": set(ids[val_end:].tolist()),
    }


def _reconstruct_split_graph_records(graph_records, positive_edges, relation_dim):
    edges_by_graph = defaultdict(list)
    for edge in positive_edges:
        edges_by_graph[int(edge["graph_id"])].append(edge)

    split_graph_records = []
    for graph in graph_records:
        graph_id = int(graph["graph_id"])
        edges = edges_by_graph.get(graph_id, [])
        if not edges:
            continue

        rel_type = np.array([int(edge["relation"]) for edge in edges], dtype=np.int64)
        edge_feat = np.zeros((len(edges), relation_dim), dtype=np.float32)
        edge_feat[np.arange(len(edges)), rel_type] = 1.0

        split_graph_records.append(
            {
                "graph_id": graph_id,
                "plate": graph["plate"],
                "cell_line": graph["cell_line"],
                "num_nodes": int(graph["node_embeddings"].shape[0]),
                "node_feat": graph["node_embeddings"].astype(np.float32),
                "node_drugs": graph["node_drugs"],
                "src": np.array([int(edge["src"]) for edge in edges], dtype=np.int64),
                "dst": np.array([int(edge["dst"]) for edge in edges], dtype=np.int64),
                "rel_type": rel_type,
                "edge_feat": edge_feat,
            }
        )

    return split_graph_records


def build_edge_prediction_splits(
    graph_records,
    positive_edges,
    relation_dim,
    seed,
    train_ratio,
    val_ratio,
    test_ratio,
    split_mode="drug_graph",
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if split_mode not in {"drug", "graph", "drug_graph"}:
        raise ValueError("split_mode must be one of: drug, graph, drug_graph")

    rng = np.random.default_rng(seed)
    graph_by_id = {int(g["graph_id"]): g for g in graph_records}

    drug_splits = _partition_ids(np.arange(relation_dim), rng, train_ratio, val_ratio)
    graph_splits = _partition_ids(
        [int(graph["graph_id"]) for graph in graph_records],
        rng,
        train_ratio,
        val_ratio,
    )

    split_raw = {"train": [], "val": [], "test": []}
    positive_set_by_graph = defaultdict(set)

    for e in positive_edges:
        gid = int(e["graph_id"])
        rel = int(e["relation"])
        positive_set_by_graph[gid].add((int(e["src"]), rel, int(e["dst"])))

        assigned_split = None
        if split_mode == "drug":
            for split_name in ["train", "val", "test"]:
                if rel in drug_splits[split_name]:
                    assigned_split = split_name
                    break
        elif split_mode == "graph":
            for split_name in ["train", "val", "test"]:
                if gid in graph_splits[split_name]:
                    assigned_split = split_name
                    break
        else:
            for split_name in ["train", "val", "test"]:
                if rel in drug_splits[split_name] and gid in graph_splits[split_name]:
                    assigned_split = split_name
                    break

        if assigned_split is not None:
            split_raw[assigned_split].append(e)

    # Serialize positives and per-split negatives.
    split_data = {}
    split_positive_edges = {}

    for split_name in ["train", "val", "test"]:
        pos_edges = split_raw[split_name]
        neg_edges = []

        for e in pos_edges:
            g = graph_by_id[int(e["graph_id"])]
            neg = _negative_sample_for_edge(e, g, positive_set_by_graph[int(e["graph_id"])])
            if neg:
                neg_edges.append(neg)

        split_positive_edges[split_name] = pos_edges
        split_data[split_name] = {
            "positive": _serialize_edges(pos_edges, relation_dim, label=1),
            "negative": _serialize_edges(neg_edges, relation_dim, label=0),
        }

    split_graph_records = {
        split_name: _reconstruct_split_graph_records(
            graph_records,
            split_positive_edges[split_name],
            relation_dim,
        )
        for split_name in ["train", "val", "test"]
    }

    return split_data, split_graph_records


def to_dgl_ready_payload(graph_records, drug_to_idx, edge_splits, split_graph_records, seed, split_cfg):
    return {
        "graphs_all": split_graph_records["train"],
        "graphs_train": split_graph_records["train"],
        "graphs_by_split": split_graph_records,
        "edge_splits": edge_splits,
        "relation_to_idx": drug_to_idx,
        "idx_to_relation": {idx: rel for rel, idx in drug_to_idx.items()},
        "seed": int(seed),
        "split_config": split_cfg,
    }


def try_save_native_dgl(graph_records, drug_to_idx, dgl_output_path: Path):
    try:
        import dgl
        import torch
    except Exception:
        return False

    dgl_output_path.parent.mkdir(parents=True, exist_ok=True)
    dgl_graphs = []
    plate_labels = []
    cell_line_labels = []

    for rec in graph_records:
        src = torch.from_numpy(rec["edge_index"][0].astype(np.int64))
        dst = torch.from_numpy(rec["edge_index"][1].astype(np.int64))
        num_nodes = rec["node_embeddings"].shape[0]
        graph = dgl.graph((src, dst), num_nodes=num_nodes)
        graph.ndata["feat"] = torch.from_numpy(rec["node_embeddings"].astype(np.float32))
        graph.edata["feat"] = torch.from_numpy(rec["edge_attr"].astype(np.float32))
        graph.edata["rel_type"] = torch.tensor(
            [drug_to_idx[d] for d in rec["edge_drugs"]], dtype=torch.int64
        )
        dgl_graphs.append(graph)
        plate_labels.append(rec["plate"])
        cell_line_labels.append(rec["cell_line"])

    dgl.save_graphs(
        str(dgl_output_path),
        dgl_graphs,
        labels={
            "plate": np.array(plate_labels, dtype=object),
            "cell_line": np.array(cell_line_labels, dtype=object),
        },
    )
    return True


def _project_to_2d(vectors: np.ndarray):
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32)


def visualize_first_nodes(graph_records, fig_path: Path, max_nodes: int = 500):
    if not graph_records:
        raise ValueError("No graph records available for visualization.")

    embeddings = []
    is_center = []
    edges_global = []
    offset = 0

    for rec in graph_records:
        node_emb = rec["node_embeddings"]
        n_nodes = node_emb.shape[0]
        embeddings.append(node_emb)
        is_center.extend([True] + [False] * (n_nodes - 1))

        src = rec["edge_index"][0] + offset
        dst = rec["edge_index"][1] + offset
        edges_global.extend(zip(src.tolist(), dst.tolist()))
        offset += n_nodes

    all_embeddings = np.vstack(embeddings).astype(np.float32)
    num_plot_nodes = min(max_nodes, all_embeddings.shape[0])
    if num_plot_nodes < 2:
        raise ValueError("Need at least 2 nodes to visualize.")

    projected = _project_to_2d(all_embeddings[:num_plot_nodes])
    center_mask = np.array(is_center[:num_plot_nodes], dtype=bool)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))

    for src, dst in edges_global:
        if src < num_plot_nodes and dst < num_plot_nodes:
            x1, y1 = projected[src]
            x2, y2 = projected[dst]
            plt.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.4, alpha=0.5, zorder=1)

    plt.scatter(
        projected[~center_mask, 0],
        projected[~center_mask, 1],
        s=10,
        alpha=0.8,
        label="Perturbed",
        zorder=2,
    )
    if np.any(center_mask):
        plt.scatter(
            projected[center_mask, 0],
            projected[center_mask, 1],
            s=25,
            alpha=0.95,
            label="DMSO centers",
            zorder=3,
        )

    plt.title(f"Star-Graph Node Projection (first {num_plot_nodes} nodes)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def save_outputs(
    graph_records,
    drug_to_idx,
    edge_splits,
    split_graph_records,
    output_path: Path,
    meta_path: Path,
    gnn_output_path: Path,
    splits_output_path: Path,
    dgl_output_path: Path,
    fig_path: Path,
    seed: int,
    split_cfg,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(graph_records, f)

    gnn_payload = to_dgl_ready_payload(
        graph_records,
        drug_to_idx,
        edge_splits,
        split_graph_records,
        seed,
        split_cfg,
    )
    with gnn_output_path.open("wb") as f:
        pickle.dump(gnn_payload, f)

    splits_json = {}
    for split_name in ["train", "val", "test"]:
        split_json = {}
        for bucket in ["positive", "negative"]:
            data = edge_splits[split_name][bucket]
            split_json[bucket] = {
                "graph_id": data["graph_id"].tolist(),
                "src": data["src"].tolist(),
                "dst": data["dst"].tolist(),
                "relation": data["relation"].tolist(),
                "label": data["label"].tolist(),
                "plate": data["plate"].tolist(),
                "cell_line": data["cell_line"].tolist(),
            }
        splits_json[split_name] = split_json

    splits_output_path.write_text(json.dumps(splits_json))

    dgl_saved = try_save_native_dgl(graph_records, drug_to_idx, dgl_output_path)
    visualize_first_nodes(graph_records, fig_path, max_nodes=500)

    metadata = {
        "num_graphs": len(graph_records),
        "num_edge_drugs": len(drug_to_idx),
        "edge_split_counts": {
            split: {
                "positive": int(edge_splits[split]["positive"]["label"].shape[0]),
                "negative": int(edge_splits[split]["negative"]["label"].shape[0]),
            }
            for split in ["train", "val", "test"]
        },
        "split_graph_counts": {
            split: len(split_graph_records[split]) for split in ["train", "val", "test"]
        },
        "edge_drug_to_index": drug_to_idx,
        "notes": "Each graph is star-shaped per (plate, cell_line) with bidirectional edges.",
        "outputs": {
            "graph_records_pickle": str(output_path),
            "gnn_payload_pickle": str(gnn_output_path),
            "edge_splits_json": str(splits_output_path),
            "dgl_binary": str(dgl_output_path) if dgl_saved else None,
            "figure": str(fig_path),
        },
        "seed": int(seed),
        "split_config": split_cfg,
        "dgl_binary_saved": dgl_saved,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Build star-shaped state-transition graphs from condition embeddings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tahoe_embeddings_parquet*.npz"),
        help=(
            "Path/pattern to embeddings NPZ where keys are (cell_line, drug, plate). "
            "Defaults to parquet NPZ prefix shards in data/ (continuous is deprecated)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/star_graphs.pkl"),
        help="Output pickle path for graph records",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/star_graphs_meta.json"),
        help="Output JSON metadata path",
    )
    parser.add_argument(
        "--gnn-output",
        type=Path,
        default=Path("data/star_graphs_gnn.pkl"),
        help="Output pickle path with DGL-ready graph payload and triplets",
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=Path("data/star_graph_edge_splits.json"),
        help="Output JSON path for edge prediction train/val/test splits",
    )
    parser.add_argument(
        "--dgl-output",
        type=Path,
        default=Path("data/star_graphs_dgl.bin"),
        help="Output DGL binary path (saved only if dgl is installed)",
    )
    parser.add_argument(
        "--fig-output",
        type=Path,
        default=Path("figs/star_graph_first_500_nodes.png"),
        help="Output figure path for first 500 nodes",
    )
    parser.add_argument(
        "--center-drugs",
        type=str,
        default="DMSO_TF",
        help="Comma-separated drug names used as center baseline (default: DMSO_TF)",
    )
    parser.add_argument(
        "--target-drug-csv",
        type=Path,
        default=Path("tahoe_to_primekg_map.csv"),
        help="CSV with 'drug' column used to define one-hot edge feature space",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for edge split and negative sampling")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio for edge prediction")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio for edge prediction")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio for edge prediction")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="graph",
        choices=["drug", "graph", "drug_graph"],
        help=(
            "How to isolate train/val/test. "
            "'graph' is the default for closed-set classification because all label classes can still appear in train. "
            "'drug' keeps unseen drugs, and 'drug_graph' enforces both unseen drugs and unseen graphs for zero-shot settings."
        ),
    )

    args = parser.parse_args()

    input_paths = resolve_input_paths(args.input)

    center_drugs = {d.strip() for d in args.center_drugs.split(",") if d.strip()}
    if not center_drugs:
        raise ValueError("No valid center drugs provided.")

    split_cfg = {
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "split_mode": args.split_mode,
    }

    condition_means = load_condition_means(input_paths)
    if not condition_means:
        raise ValueError(
            "No valid condition embeddings found in input NPZ. "
            "Expected tuple keys like (cell_line, drug, plate)."
        )

    grouped_conditions = build_grouped_conditions(condition_means)
    drug_to_idx = load_drug_mapping(args.target_drug_csv, center_drugs)
    graph_records, positive_edges = build_star_graphs(grouped_conditions, center_drugs, drug_to_idx)

    if not graph_records:
        raise ValueError(
            "No star graphs produced. Ensure center drug exists (default: DMSO_TF) and perturbed drugs are present."
        )

    edge_splits, split_graph_records = build_edge_prediction_splits(
        graph_records,
        positive_edges,
        relation_dim=len(drug_to_idx),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
    )

    save_outputs(
        graph_records,
        drug_to_idx,
        edge_splits,
        split_graph_records,
        args.output,
        args.meta,
        args.gnn_output,
        args.splits_output,
        args.dgl_output,
        args.fig_output,
        args.seed,
        split_cfg,
    )

    if len(input_paths) == 1:
        print(f"Input embeddings: {input_paths[0]}")
    else:
        print(f"Input embeddings shards: {len(input_paths)} files")
        print(f"First shard: {input_paths[0]}")
        print(f"Last shard: {input_paths[-1]}")
    print(f"Target drug CSV: {args.target_drug_csv}")
    print(f"Saved {len(graph_records)} graphs -> {args.output}")
    print(f"Saved DGL-ready payload -> {args.gnn_output}")
    print(f"Saved edge splits -> {args.splits_output}")
    print(f"Saved figure -> {args.fig_output}")
    print(f"Saved metadata -> {args.meta}")
    print(f"Edge drug classes: {len(drug_to_idx)}")


if __name__ == "__main__":
    main()