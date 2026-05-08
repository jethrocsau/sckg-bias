import json
import pickle

import numpy as np
import pandas as pd


def _edge_keys(split_bucket):
    return {
        (
            int(graph_id),
            int(src),
            int(relation),
            int(dst),
        )
        for graph_id, src, relation, dst in zip(
            split_bucket["graph_id"],
            split_bucket["src"],
            split_bucket["relation"],
            split_bucket["dst"],
        )
    }


def _relation_set(split_bucket):
    return {int(relation) for relation in split_bucket["relation"]}


def _graph_set(split_bucket):
    return {int(graph_id) for graph_id in split_bucket["graph_id"]}


def _lookup_split_graphs(payload, split_name):
    graphs_by_split = payload.get("graphs_by_split")
    if isinstance(graphs_by_split, dict) and split_name in graphs_by_split:
        graphs = graphs_by_split[split_name]
    elif split_name == "train":
        graphs = payload.get("graphs_train", payload.get("graphs_all", []))
    else:
        graphs = payload.get("graphs_all", [])
    return {int(graph["graph_id"]): graph for graph in graphs}


def _center_relation_signature(graph):
    src = np.asarray(graph["src"], dtype=np.int64)
    dst = np.asarray(graph["dst"], dtype=np.int64)
    rel = np.asarray(graph["rel_type"], dtype=np.int64)
    return frozenset(int(r) for s, d, r in zip(src, dst, rel) if int(s) == 0 and int(d) != 0)


def check_leakage(payload_path, edge_splits_path):
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)

    with open(edge_splits_path, "r") as f:
        splits = json.load(f)

    train_pos = splits["train"]["positive"]
    val_pos = splits["val"]["positive"]
    test_pos = splits["test"]["positive"]

    split_mode = payload.get("split_config", {}).get("split_mode", "unknown")

    overlaps = {
        "train_val_edge_overlap": len(_edge_keys(train_pos) & _edge_keys(val_pos)),
        "train_test_edge_overlap": len(_edge_keys(train_pos) & _edge_keys(test_pos)),
        "val_test_edge_overlap": len(_edge_keys(val_pos) & _edge_keys(test_pos)),
        "train_val_relation_overlap": len(_relation_set(train_pos) & _relation_set(val_pos)),
        "train_test_relation_overlap": len(_relation_set(train_pos) & _relation_set(test_pos)),
        "val_test_relation_overlap": len(_relation_set(val_pos) & _relation_set(test_pos)),
        "train_val_graph_overlap": len(_graph_set(train_pos) & _graph_set(val_pos)),
        "train_test_graph_overlap": len(_graph_set(train_pos) & _graph_set(test_pos)),
        "val_test_graph_overlap": len(_graph_set(val_pos) & _graph_set(test_pos)),
    }

    train_graphs = _lookup_split_graphs(payload, "train")
    val_graphs = _lookup_split_graphs(payload, "val")
    test_graphs = _lookup_split_graphs(payload, "test")

    signature_rows = []
    for left_name, right_name, left_graphs, right_graphs in [
        ("train", "val", train_graphs, val_graphs),
        ("train", "test", train_graphs, test_graphs),
        ("val", "test", val_graphs, test_graphs),
    ]:
        common_graph_ids = sorted(set(left_graphs) & set(right_graphs))
        for graph_id in common_graph_ids:
            left_sig = _center_relation_signature(left_graphs[graph_id])
            right_sig = _center_relation_signature(right_graphs[graph_id])
            union = len(left_sig | right_sig)
            jaccard = (len(left_sig & right_sig) / union) if union else 0.0
            signature_rows.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "graph_id": graph_id,
                    "jaccard_relation_overlap": jaccard,
                    "left_relations": len(left_sig),
                    "right_relations": len(right_sig),
                }
            )

    print("Split mode:", split_mode)
    for name, value in overlaps.items():
        print(f"{name}: {value}")

    overlap_df = pd.DataFrame(signature_rows)
    if not overlap_df.empty:
        overlap_df.to_csv("neighborhood_leakage_report.csv", index=False)
        print(
            "Saved neighborhood relation-overlap report -> neighborhood_leakage_report.csv"
        )
        print(
            "Mean shared-graph neighborhood overlap:",
            f"{overlap_df['jaccard_relation_overlap'].mean():.4f}",
        )

    required_zero_keys = [
        "train_val_edge_overlap",
        "train_test_edge_overlap",
        "val_test_edge_overlap",
    ]
    if split_mode == "graph":
        required_zero_keys.extend([
            "train_val_graph_overlap",
            "train_test_graph_overlap",
            "val_test_graph_overlap",
        ])
    elif split_mode == "drug":
        required_zero_keys.extend([
            "train_val_relation_overlap",
            "train_test_relation_overlap",
            "val_test_relation_overlap",
        ])
    elif split_mode == "drug_graph":
        required_zero_keys.extend([
            "train_val_graph_overlap",
            "train_test_graph_overlap",
            "val_test_graph_overlap",
            "train_val_relation_overlap",
            "train_test_relation_overlap",
            "val_test_relation_overlap",
        ])

    if any(overlaps[key] > 0 for key in required_zero_keys):
        print("\nALERT: train/val/test are not disjoint for the selected split mode.")
    else:
        print("\nSUCCESS: No forbidden overlap detected for the selected split mode.")


if __name__ == "__main__":
    check_leakage("data/star_graphs_gnn.pkl", "data/star_graph_edge_splits.json")