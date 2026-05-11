"""Entrypoint scaffold for MoE StarFactor pipeline with Interpretability Exports."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchinfo import summary

from pipeline.models.baselines import ContextOnlyBaseline
from pipeline.models.baselines import ContextPlusKGBaseline
from pipeline.models.baselines import H0OnlyBaseline
from pipeline.models.baselines import KGContextBaseline
from pipeline.models.baselines import TargetPlusContextBaseline
from pipeline.models.baselines import TargetPlusContextKGBaseline
from pipeline.models.baselines import TargetPlusKGBaseline
from pipeline.models.baselines import TargetOnlyBaseline
from pipeline.models.star_factor import StarFactorModel
from pipeline.primekg_bridge import PrimeKGBridgeConfig
from pipeline.primekg_bridge import load_drug_class_order
from pipeline.primekg_bridge import load_entity_id_to_name
from pipeline.primekg_bridge import load_multirel_relation_context
from pipeline.primekg_bridge import load_multirel_relation_ids
from pipeline.primekg_bridge import load_multirel_relation_mask
from pipeline.primekg_bridge import load_relation_id_to_name
from pipeline.primekg_bridge import precompute_h_kg_paths
from pipeline.primekg_bridge import precompute_h_kg_multihop
from pipeline.star_data import StarDataConfig
from pipeline.star_data import build_train_epoch_sampler
from pipeline.star_data import build_split_samples
from pipeline.star_data import collate_samples
from pipeline.star_data import load_edge_splits
from pipeline.train_eval import TrainEvalConfig
from pipeline.train_eval import train_step

@dataclass
class StarFactorConfig:
    num_classes: int = 188
    input_dim: int = 512
    d_model: int = 512
    d_kg: int = 256
    d_msg: int = 128
    gat_depth: int = 1
    num_kg_relations: int = 1
    kg_enabled: bool = True
    num_factors: int = 4
    dropout: float = 0.3
    model_variant: str = "full"
    kg_embed_method: str = "mean_decay"
    kg_subtree_pool_method: str = "match"
    kg_relation_pool_method: str = "match"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoE StarFactor pipeline.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precompute-kg-multihop", action="store_true")
    parser.add_argument("--kg-hops", type=int, default=3)
    parser.add_argument(
        "--kg-embed-method",
        type=str,
        default="mean_decay",
        choices=["mean_decay", "gat", "dgl_gat", "path_attn"],
        help="For relation-based KG context, this now controls how each immediate-relation subtree is summarized before relation selection. `dgl_gat` is kept as a backward-compatible alias for `gat`.",
    )
    parser.add_argument(
        "--kg-subtree-pool-method",
        type=str,
        default="match",
        choices=["match", "mean_decay", "gat"],
        help="Within-relation subtree summarizer. `match` reuses --kg-embed-method for relation-based KG inputs.",
    )
    parser.add_argument(
        "--kg-relation-pool-method",
        type=str,
        default="match",
        choices=["match", "mean_decay", "gat"],
        help="How the per-relation rows are pooled after subtree summarization. `match` uses deterministic mean pooling for `mean_decay` and attention pooling for `gat`/`dgl_gat`.",
    )
    parser.add_argument("--kg-path-max-paths", type=int, default=32)
    parser.add_argument("--primekg-dir", type=Path, default=Path("data/primekgpp_grace_redaf"))
    parser.add_argument("--star-meta-path", type=Path, default=Path("data/star_graphs_meta.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-embed-dim", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--gat-depth", type=int, default=1)
    parser.add_argument("--gat-hidden-dim", type=int, default=128)
    parser.add_argument("--num-factors", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--model-variant",
        type=str,
        default="full",
        choices=["full", "h0_only", "target_only", "context_only", "kg_context", "target_plus_context", "context_plus_kg", "target_plus_kg", "target_plus_context_kg"],
        help="Which model to train: full graph model or one of the diagnostic baselines.",
    )
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-div", type=float, default=1e-4)
    parser.add_argument("--lambda-sp", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--analysis-output-dir", type=Path, default=Path("results/star_factor_analysis"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kg-relation-limit", type=int, default=0)

    # --- ADDED ARGUMENTS TO FIX THE ERROR ---
    parser.add_argument(
        "--selection-metric", 
        type=str, 
        default="val/loss_ce", 
        help="Metric used to determine the best model checkpoint."
    )
    parser.add_argument(
        "--early-stopping-patience", 
        type=int, 
        default=10, 
        help="Number of epochs to wait for improvement before stopping."
    )
    parser.add_argument(
        "--early-stopping-min-delta", 
        type=float, 
        default=0.001, 
        help="Minimum change to qualify as an improvement."
    )
    
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _args_to_record(args: argparse.Namespace) -> dict[str, object]:
    return {key: _jsonable(value) for key, value in vars(args).items()}

def build_model(device: str, args: argparse.Namespace, stage: int, num_kg_relations: int = 1, d_kg: int = 256) -> StarFactorModel:
    cfg = StarFactorConfig(
        input_dim=args.input_embed_dim,
        d_model=args.embedding_size,
        d_msg=args.gat_hidden_dim,
        gat_depth=args.gat_depth,
        d_kg=d_kg,
        num_kg_relations=num_kg_relations,
        kg_enabled=(stage == 2),
        num_factors=args.num_factors,
        dropout=args.dropout,
        model_variant=args.model_variant,
        kg_embed_method=args.kg_embed_method,
        kg_subtree_pool_method=args.kg_subtree_pool_method,
        kg_relation_pool_method=args.kg_relation_pool_method,
    )
    if args.model_variant == "h0_only":
        return H0OnlyBaseline(cfg).to(device)
    if args.model_variant == "target_only":
        return TargetOnlyBaseline(cfg).to(device)
    if args.model_variant == "context_only":
        return ContextOnlyBaseline(cfg).to(device)
    if args.model_variant == "kg_context":
        return KGContextBaseline(cfg).to(device)
    if args.model_variant == "target_plus_context":
        return TargetPlusContextBaseline(cfg).to(device)
    if args.model_variant == "context_plus_kg":
        return ContextPlusKGBaseline(cfg).to(device)
    if args.model_variant == "target_plus_kg":
        return TargetPlusKGBaseline(cfg).to(device)
    if args.model_variant == "target_plus_context_kg":
        return TargetPlusContextKGBaseline(cfg).to(device)
    return StarFactorModel(cfg).to(device)


def _validate_closed_set_split(cfg: StarDataConfig) -> None:
    edge_splits = load_edge_splits(cfg.edge_splits_path)
    split_mode = edge_splits.get("split_config", {}).get("split_mode")
    if split_mode is None:
        try:
            import pickle

            with cfg.gnn_payload_path.open("rb") as f:
                payload = pickle.load(f)
            split_mode = payload.get("split_config", {}).get("split_mode")
        except Exception:
            split_mode = None

    train_rel = set(edge_splits["train"]["positive"]["relation"])
    val_rel = set(edge_splits["val"]["positive"]["relation"])
    test_rel = set(edge_splits["test"]["positive"]["relation"])
    unseen_val = sorted(val_rel - train_rel)
    unseen_test = sorted(test_rel - train_rel)

    if unseen_val or unseen_test:
        raise ValueError(
            "Validation/test contain label classes unseen during training. "
            f"split_mode={split_mode or 'unknown'} gives {len(unseen_val)} unseen val classes and "
            f"{len(unseen_test)} unseen test classes. "
            "This model is a closed-set classifier over fixed class IDs, so use graph-only splits "
            "when training it, e.g. rebuild data with --split-mode graph. "
            "Keep drug/drug_graph only for zero-shot experiments with a different objective."
        )

def _set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior in some CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def _iter_batches(samples, batch_size, device, shuffle, seed):
    indices = np.arange(len(samples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_samples = [samples[int(i)] for i in batch_idx]
        yield collate_samples(batch_samples, device=device)

@torch.no_grad()
def _evaluate(model, samples, batch_size, device, h_kg_multi, return_details=False, desc="Evaluating"):
    model.eval()
    if not samples:
        return {"loss/ce": 0.0, "acc": 0.0}

    total_loss, total_count = 0.0, 0
    all_true, all_pred, all_conf, all_gates = [], [], [], []
    sample_metadata = [] # To store IDs and neighbor classes for graph reconstruction

    n_batches = (len(samples) + batch_size - 1) // batch_size
    pbar = tqdm(_iter_batches(samples, batch_size, device, shuffle=False, seed=0), 
                total=n_batches, desc=desc, leave=False)

    for batch in pbar:
        logits, aux_data = model(
            h0=batch["h0"],
            h_known=batch["h_known"],
            y_known=batch["y_known"],
            h_target=batch["h_target"],
            known_mask=batch.get("known_mask"),
            h_kg_multi=h_kg_multi
        )
        labels = batch["y"]
        bsz = labels.shape[0]
        
        ce = F.cross_entropy(logits, labels)
        total_loss += float(ce.detach().cpu()) * bsz
        total_count += bsz

        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
        all_true.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        
        if return_details:
            all_conf.append(conf.cpu().numpy())
            all_gates.append(aux_data["gate_weights"].cpu().numpy())
            # Capture the "Resolution" metadata
            sample_metadata.append({
                "h0_ids": batch["h0_ids"].cpu().numpy(),
                "y_known": batch["y_known"].cpu().numpy(),
                "correct": (pred == labels).cpu().numpy()
            })

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc = float((y_true == y_pred).mean())

    return {
        "loss/ce": total_loss / max(total_count, 1),
        "acc": acc,
        "y_true": y_true if return_details else None,
        "y_pred": y_pred if return_details else None,
        "y_conf": np.concatenate(all_conf) if return_details else None,
        "gate_weights": np.concatenate(all_gates) if return_details else None,
        "sample_metadata": sample_metadata if return_details else None
    }

def _export_prediction_details(eval_results, class_names, split_name, out_dir, run_name):
    """Exports sample-level predictions and MoE gate selections."""
    rows = []
    y_true = eval_results["y_true"]
    y_pred = eval_results["y_pred"]
    y_conf = eval_results["y_conf"]
    gates = eval_results["gate_weights"] # [N, F]
    
    num_factors = gates.shape[1]
    
    for i in range(len(y_true)):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        
        row = {
            "sample_id": i,
            "split": split_name,
            "true_edge_class_idx": true_idx,
            "true_edge_name": class_names[true_idx] if true_idx < len(class_names) else f"class_{true_idx}",
            "pred_edge_class_idx": pred_idx,
            "pred_edge_name": class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}",
            "correct": bool(true_idx == pred_idx),
            "confidence": float(y_conf[i]),
            "dominant_factor_idx": int(np.argmax(gates[i]))
        }
        
        # Log individual gate weights
        for f in range(num_factors):
            row[f"gate_weight_factor_{f}"] = float(gates[i, f])
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}.{split_name}_predictions.csv", index=False)


def _export_prediction_comparison(val_details, test_details, class_names, out_dir, run_name):
    split_tables = []
    summary_rows = []

    for split_name, eval_results in [("val", val_details), ("test", test_details)]:
        y_true = eval_results["y_true"]
        y_pred = eval_results["y_pred"]
        y_conf = eval_results["y_conf"]
        if y_true is None or y_pred is None or y_conf is None:
            continue

        df = pd.DataFrame(
            {
                "split": split_name,
                "true_edge_class_idx": y_true.astype(int),
                "pred_edge_class_idx": y_pred.astype(int),
                "confidence": y_conf.astype(float),
            }
        )
        df["true_edge_name"] = df["true_edge_class_idx"].map(
            lambda idx: class_names[idx] if idx < len(class_names) else f"class_{idx}"
        )
        df["pred_edge_name"] = df["pred_edge_class_idx"].map(
            lambda idx: class_names[idx] if idx < len(class_names) else f"class_{idx}"
        )
        df["correct"] = df["true_edge_class_idx"] == df["pred_edge_class_idx"]

        grouped = (
            df.groupby(
                ["split", "true_edge_class_idx", "true_edge_name", "pred_edge_class_idx", "pred_edge_name"],
                as_index=False,
            )
            .agg(
                count=("confidence", "size"),
                confidence_mean=("confidence", "mean"),
                confidence_std=("confidence", "std"),
                correct_rate=("correct", "mean"),
            )
        )
        split_tables.append(grouped)

        summary_rows.append(
            {
                "split": split_name,
                "num_samples": int(len(df)),
                "acc": float(df["correct"].mean()),
                "confidence_mean": float(df["confidence"].mean()),
                "confidence_correct_mean": float(df.loc[df["correct"], "confidence"].mean()) if df["correct"].any() else np.nan,
                "confidence_incorrect_mean": float(df.loc[~df["correct"], "confidence"].mean()) if (~df["correct"]).any() else np.nan,
                "num_unique_true": int(df["true_edge_class_idx"].nunique()),
                "num_unique_pred": int(df["pred_edge_class_idx"].nunique()),
            }
        )

    if split_tables:
        pd.concat(split_tables, ignore_index=True).to_csv(out_dir / f"{run_name}.prediction_comparison.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / f"{run_name}.prediction_summary.csv", index=False)

def _export_factor_analysis(model, relation_ids, relation_names, class_names, out_dir, run_name):
    """Exports the learned factor weights mapping classes to KG relations."""
    if not hasattr(model, 'kg_enabled') or not model.kg_enabled:
        return
        
    # Apply softmax to get the actual routing distribution [F, C, R]
    alpha_rel = torch.softmax(model.factor_weights, dim=-1).detach().cpu().numpy()
    num_factors, num_classes, num_relations = alpha_rel.shape
    
    rows = []
    for f in range(num_factors):
        for c in range(num_classes):
            class_name = class_names[c] if c < len(class_names) else f"class_{c}"
            
            # To keep the CSV manageable, we only export the top 5 relations per factor/class
            # or all if R < 5
            top_r_indices = np.argsort(alpha_rel[f, c, :])[::-1][:min(5, num_relations)]
            
            for r_idx in top_r_indices:
                rel_id = relation_ids[r_idx] if relation_ids and r_idx < len(relation_ids) else r_idx
                rel_name = relation_names[r_idx] if relation_names and r_idx < len(relation_names) else f"rel_{rel_id}"
                
                rows.append({
                    "factor_idx": f,
                    "class_idx": c,
                    "class_name": class_name,
                    "relation_idx": r_idx,
                    "relation_id": rel_id,
                    "relation_name": rel_name,
                    "weight": float(alpha_rel[f, c, r_idx])
                })
                
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}.factor_kg_mappings.csv", index=False)


def _move_kg_context_to_device(h_kg_multi, device: str):
    if h_kg_multi is None:
        return None
    if isinstance(h_kg_multi, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in h_kg_multi.items()
        }
    return h_kg_multi.to(device)


def _export_relation_weight_biases(model, h_kg_multi, relation_ids, relation_names, class_names, out_dir, run_name, bridge_cfg=None):
    if not hasattr(model, "export_relation_weights") or h_kg_multi is None:
        return
    relation_weights = model.export_relation_weights(h_kg_multi)
    if relation_weights is None:
        return

    # Load entity and relation mappings for all exports
    entity_map = load_entity_id_to_name(bridge_cfg.entity2id_path) if bridge_cfg and bridge_cfg.entity2id_path.exists() else {}
    rel_map = load_relation_id_to_name(bridge_cfg.relation2id_path) if bridge_cfg and bridge_cfg.relation2id_path.exists() else {}

    if isinstance(h_kg_multi, dict) and "path_mask" in h_kg_multi:
        path_mask = h_kg_multi["path_mask"].detach().cpu().numpy().astype(bool, copy=False)
        endpoint_ids = h_kg_multi["endpoint_ids"].detach().cpu().numpy()
        path_hops = h_kg_multi["path_hops"].detach().cpu().numpy()
        relation_seqs = h_kg_multi["relation_seqs"].detach().cpu().numpy()
        node_seqs = h_kg_multi["node_seqs"].detach().cpu().numpy()

        rows = []
        for class_idx in range(relation_weights.shape[0]):
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            for path_idx in range(relation_weights.shape[1]):
                if not path_mask[class_idx, path_idx]:
                    continue

                rel_seq = [int(rel_id) for rel_id in relation_seqs[class_idx, path_idx] if int(rel_id) >= 0]
                node_seq = [int(node_id) for node_id in node_seqs[class_idx, path_idx] if int(node_id) >= 0]
                rows.append(
                    {
                        "class_idx": class_idx,
                        "class_name": class_name,
                        "path_idx": path_idx,
                        "endpoint_id": int(endpoint_ids[class_idx, path_idx]),
                        "endpoint_name": entity_map.get(int(endpoint_ids[class_idx, path_idx]), f"entity_{int(endpoint_ids[class_idx, path_idx])}"),
                        "path_hops": int(path_hops[class_idx, path_idx]),
                        "relation_seq": " | ".join(rel_map.get(rel_id, f"relation_{rel_id}") for rel_id in rel_seq),
                        "node_seq": " | ".join(entity_map.get(node_id, f"entity_{node_id}") for node_id in node_seq),
                        "weight": float(relation_weights[class_idx, path_idx]),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{run_name}.kg_path_weights.csv", index=False)
        summary = (
            df.groupby(["path_idx", "path_hops", "relation_seq"], as_index=False)
            .agg(
                weight_mean=("weight", "mean"),
                weight_std=("weight", "std"),
                weight_min=("weight", "min"),
                weight_max=("weight", "max"),
            )
        )
        summary.to_csv(out_dir / f"{run_name}.kg_path_weight_summary.csv", index=False)
        return

    # Extract neighbor data from h_kg_multi dict BEFORE potentially converting to tensor
    member_node_ids_flat = None
    member_offsets = None
    member_weights_flat = None
    if isinstance(h_kg_multi, dict):
        member_node_ids_flat = h_kg_multi.get("member_node_ids_flat")
        member_offsets = h_kg_multi.get("member_offsets")
        member_weights_flat = h_kg_multi.get("member_weights_flat")
        if "rel_embeddings" in h_kg_multi:
            h_kg_multi = h_kg_multi["rel_embeddings"]

    names = relation_names or []
    ids = relation_ids or []
    
    # member_node_ids_flat, member_offsets, member_weights_flat already extracted above

    rows = []
    for class_idx in range(relation_weights.shape[0]):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        for rel_idx in range(relation_weights.shape[1]):
            rel_name = names[rel_idx] if rel_idx < len(names) else f"relation_{rel_idx}"
            rel_id = ids[rel_idx] if rel_idx < len(ids) else rel_idx
            base_row = {
                "class_idx": class_idx,
                "class_name": class_name,
                "relation_idx": rel_idx,
                "relation_id": rel_id,
                "relation_name": rel_name,
                "weight": float(relation_weights[class_idx, rel_idx]),
            }
            
            # Add neighbor information if available
            if member_offsets is not None and member_node_ids_flat is not None:
                try:
                    start_idx = int(member_offsets[class_idx, rel_idx, 0].item())
                    end_idx = int(member_offsets[class_idx, rel_idx, 1].item())
                    neighbor_ids = member_node_ids_flat[start_idx:end_idx]
                    neighbor_names = [
                        entity_map.get(int(nid), f"entity_{int(nid)}")
                        for nid in neighbor_ids
                    ]
                    neighbor_weights = (
                        member_weights_flat[start_idx:end_idx].cpu().numpy().tolist()
                        if member_weights_flat is not None else [1.0] * len(neighbor_ids)
                    )
                    base_row["num_neighbors"] = len(neighbor_ids)
                    base_row["neighbor_ids"] = " | ".join(str(int(nid)) for nid in neighbor_ids)
                    base_row["neighbor_names"] = " | ".join(neighbor_names)
                    base_row["neighbor_weights"] = " | ".join(f"{w:.4f}" for w in neighbor_weights)
                except (IndexError, RuntimeError):
                    pass  # Skip if unable to unpack
            
            rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}.kg_relation_weights.csv", index=False)
    summary = (
        df.groupby(["relation_idx", "relation_id", "relation_name"], as_index=False)
        .agg(
            weight_mean=("weight", "mean"),
            weight_std=("weight", "std"),
            weight_min=("weight", "min"),
            weight_max=("weight", "max"),
        )
    )
    summary.to_csv(out_dir / f"{run_name}.kg_relation_weight_summary.csv", index=False)


def _export_subtree_members(h_kg_multi, relation_ids, relation_names, class_names, out_dir, run_name, bridge_cfg=None):
    if not isinstance(h_kg_multi, dict):
        return

    member_offsets = h_kg_multi.get("member_offsets")
    member_node_ids_flat = h_kg_multi.get("member_node_ids_flat")
    member_weights_flat = h_kg_multi.get("member_weights_flat")
    rel_mask = h_kg_multi.get("rel_mask")
    if member_offsets is None or member_node_ids_flat is None:
        return

    entity_map = load_entity_id_to_name(bridge_cfg.entity2id_path) if bridge_cfg and bridge_cfg.entity2id_path.exists() else {}
    names = relation_names or []
    ids = relation_ids or []

    rows = []
    num_classes = member_offsets.shape[0]
    num_rel = member_offsets.shape[1]
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        for rel_idx in range(num_rel):
            if rel_mask is not None and not bool(rel_mask[class_idx, rel_idx]):
                continue

            start_idx = int(member_offsets[class_idx, rel_idx, 0].item())
            end_idx = int(member_offsets[class_idx, rel_idx, 1].item())
            if end_idx <= start_idx:
                continue

            rel_name = names[rel_idx] if rel_idx < len(names) else f"relation_{rel_idx}"
            rel_id = ids[rel_idx] if rel_idx < len(ids) else rel_idx

            neighbor_ids = member_node_ids_flat[start_idx:end_idx]
            if member_weights_flat is not None:
                neighbor_weights = member_weights_flat[start_idx:end_idx].cpu().numpy().tolist()
            else:
                neighbor_weights = [1.0] * len(neighbor_ids)

            for idx, node_id in enumerate(neighbor_ids):
                node_id_int = int(node_id)
                rows.append(
                    {
                        "class_idx": class_idx,
                        "class_name": class_name,
                        "relation_idx": rel_idx,
                        "relation_id": rel_id,
                        "relation_name": rel_name,
                        "member_rank": idx,
                        "member_node_id": node_id_int,
                        "member_name": entity_map.get(node_id_int, f"entity_{node_id_int}"),
                        "member_weight": float(neighbor_weights[idx]),
                    }
                )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"{run_name}.kg_subtree_members.csv", index=False)

def _run_single_training(run_name, stage, args, device, relation_limit, out_dir):
    _validate_closed_set_split(StarDataConfig())

    class_names = load_drug_class_order(meta_path=args.star_meta_path)
    bridge_cfg = PrimeKGBridgeConfig(primekg_dir=args.primekg_dir)

    relation_ids, relation_names = None, None
    relation_context_for_export = None
    if stage == 1 or args.model_variant in {"h0_only", "target_only", "context_only", "target_plus_context"}:
        h_kg_multi = None
        num_kg_relations = 1
        d_kg = 256
    else:
        if args.kg_embed_method == "path_attn":
            h_kg_multi = _move_kg_context_to_device(
                precompute_h_kg_paths(
                    bridge_cfg,
                    class_names,
                    hops=args.kg_hops,
                    relation_limit=relation_limit,
                    max_paths=args.kg_path_max_paths,
                ),
                device,
            )
            num_kg_relations = h_kg_multi["path_embeddings"].shape[1]
            d_kg = h_kg_multi["path_embeddings"].shape[2]
        else:
            rel_embeddings = precompute_h_kg_multihop(
                bridge_cfg,
                class_names,
                hops=args.kg_hops,
                relation_limit=relation_limit,
            )
            relation_context = load_multirel_relation_context(bridge_cfg, hops=args.kg_hops, relation_limit=relation_limit)
            relation_context_for_export = relation_context
            relation_mask = load_multirel_relation_mask(bridge_cfg, hops=args.kg_hops, relation_limit=relation_limit)
            num_kg_relations = rel_embeddings.shape[1]
            d_kg = rel_embeddings.shape[2]

            if args.model_variant == "full":
                h_kg_multi = rel_embeddings.to(device)
            else:
                h_kg_multi = relation_context or {}
                h_kg_multi["rel_embeddings"] = rel_embeddings
                if relation_mask is not None:
                    h_kg_multi["rel_mask"] = relation_mask
                h_kg_multi = _move_kg_context_to_device(h_kg_multi, device)

            # Load metadata for interpretability
            relation_ids = load_multirel_relation_ids(bridge_cfg, hops=args.kg_hops, relation_limit=relation_limit)
            rel_map = load_relation_id_to_name(bridge_cfg.relation2id_path) if bridge_cfg.relation2id_path.exists() else {}
            relation_names = [rel_map.get(rid, f"relation_{rid}") for rid in relation_ids] if relation_ids else None

    model = build_model(device, args, stage, num_kg_relations, d_kg)

    # print model
    print(f"Model variant: {args.model_variant}")
    print(f"Model summary: {summary(model)}")
    
    train_sampler = build_train_epoch_sampler(StarDataConfig())
    val_samples = build_split_samples(StarDataConfig(), split="val")
    test_samples = build_split_samples(StarDataConfig(), split="test")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    cfg = TrainEvalConfig(
        lambda_div=args.lambda_div,
        lambda_sp=args.lambda_sp,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
    )

    run_metadata = _args_to_record(args)
    run_metadata.update(
        {
            "run_name": run_name,
            "stage": stage,
            "relation_limit": relation_limit if relation_limit is not None else "all",
            "kg_source_dir": str(args.primekg_dir),
            "kg_source_key": args.primekg_dir.name.replace("primekgpp_", ""),
            "num_kg_relations": int(num_kg_relations),
            "d_kg": int(d_kg),
        }
    )

    history_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.run_config.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True))
    
    best_val_acc = -1.0
    best_model_state = None

    # Early stopping setup
    _lower_is_better = args.selection_metric.endswith("loss_ce")
    _best_metric = float("inf") if _lower_is_better else float("-inf")
    _es_patience = args.early_stopping_patience  # 0 = disabled
    _es_min_delta = args.early_stopping_min_delta
    _es_wait = 0
    _stopped_early = False

    # Main Epoch Progress Bar
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc=f"Training {run_name}")
    for epoch in epoch_pbar:
        train_samples = train_sampler.sample(seed=args.seed + epoch)
        epoch_ce = 0.0
        count = 0
        
        # Nested Training Batch Progress Bar
        n_train_batches = (len(train_samples) + args.batch_size - 1) // args.batch_size
        train_pbar = tqdm(_iter_batches(train_samples, args.batch_size, device, True, args.seed + epoch),
                          total=n_train_batches, desc=f"Epoch {epoch} [Train]", leave=False)
        
        model.train()
        for batch in train_pbar:
            batch["h_kg_multi"] = h_kg_multi
            metrics = train_step(model, batch, optimizer, cfg, epoch=epoch)
            batch_ce = metrics["loss/ce"]
            epoch_ce += batch_ce
            count += 1
            train_pbar.set_postfix({"batch_ce": f"{batch_ce:.4f}"})
            
        # Validation with its own nested bar
        val_eval = _evaluate(model, val_samples, args.batch_size, device, h_kg_multi, desc=f"Epoch {epoch} [Val]")
        
        row = {
            "epoch": epoch,
            "train/loss_ce": epoch_ce / max(count, 1),
            "val/loss_ce": val_eval["loss/ce"],
            "val/acc": val_eval["acc"],
        }
        row.update(run_metadata)
        history_rows.append(row)

        # Determine monitored metric value
        _metric_val = row[args.selection_metric]
        _improved = (
            (_metric_val < _best_metric - _es_min_delta) if _lower_is_better
            else (_metric_val > _best_metric + _es_min_delta)
        )

        # Update Main Epoch Bar with current scores
        epoch_pbar.set_postfix({
            "tr_ce": f"{row['train/loss_ce']:.3f}", 
            "val_ce": f"{row['val/loss_ce']:.3f}", 
            "val_acc": f"{row['val/acc']:.3f}",
            "es_wait": f"{_es_wait}/{_es_patience}" if _es_patience > 0 else "off",
        })
        
        # Checkpointing
        if val_eval["acc"] > best_val_acc:
            best_val_acc = val_eval["acc"]
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Early stopping logic
        if _es_patience > 0:
            if _improved:
                _best_metric = _metric_val
                _es_wait = 0
            else:
                _es_wait += 1
                if _es_wait >= _es_patience:
                    print(f"[{run_name}] Early stopping at epoch {epoch} (no improvement in {args.selection_metric} for {_es_patience} epochs).")
                    _stopped_early = True
                    break

    # --- Post-Training Interpretability Generation ---
    stop_reason = "early stopping" if _stopped_early else f"epoch {args.epochs}"
    print(f"[{run_name}] Training complete ({stop_reason}). Best Val Acc: {best_val_acc:.4f}. Generating meta-analysis artifacts...")
    
    # 1. Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), out_dir / f"{run_name}.best_model.pt")
    pd.DataFrame(history_rows).to_csv(out_dir / f"{run_name}_history.csv", index=False)
    
    # 2. Detailed Evaluation & Sample Routing Export
    val_details = _evaluate(model, val_samples, args.batch_size, device, h_kg_multi, return_details=True)
    test_details = _evaluate(model, test_samples, args.batch_size, device, h_kg_multi, return_details=True)
    
    _export_prediction_details(val_details, class_names, "val", out_dir, run_name)
    _export_prediction_details(test_details, class_names, "test", out_dir, run_name)
    _export_prediction_comparison(val_details, test_details, class_names, out_dir, run_name)
    
    # 3. Factor-Relation Map Export
    if stage == 2:
        _export_factor_analysis(model, relation_ids, relation_names, class_names, out_dir, run_name)
    _export_relation_weight_biases(model, h_kg_multi, relation_ids, relation_names, class_names, out_dir, run_name, bridge_cfg=bridge_cfg)
    _export_subtree_members(relation_context_for_export, relation_ids, relation_names, class_names, out_dir, run_name, bridge_cfg=bridge_cfg)
    
    return {
        **run_metadata,
        "val_acc_best": float(best_val_acc),
        "test_acc": float(test_details["acc"]),
        "history_path": str(out_dir / f"{run_name}_history.csv"),
        "run_config_path": str(out_dir / f"{run_name}.run_config.json"),
        "val_predictions_path": str(out_dir / f"{run_name}.val_predictions.csv"),
        "test_predictions_path": str(out_dir / f"{run_name}.test_predictions.csv"),
        "prediction_summary_path": str(out_dir / f"{run_name}.prediction_summary.csv"),
        "prediction_comparison_path": str(out_dir / f"{run_name}.prediction_comparison.csv"),
        "kg_relation_weights_path": str(out_dir / f"{run_name}.kg_relation_weights.csv"),
        "kg_relation_weight_summary_path": str(out_dir / f"{run_name}.kg_relation_weight_summary.csv"),
        "kg_subtree_members_path": str(out_dir / f"{run_name}.kg_subtree_members.csv"),
        "kg_path_weights_path": str(out_dir / f"{run_name}.kg_path_weights.csv"),
        "kg_path_weight_summary_path": str(out_dir / f"{run_name}.kg_path_weight_summary.csv"),
    }

def main():
    args = parse_args()
    _set_seed(args.seed)
    
    if args.run_training:
        out_dir = args.analysis_output_dir
        summaries = []
        relation_limit = args.kg_relation_limit if args.kg_relation_limit > 0 else None
        
        summaries.append(
            _run_single_training("single_run", args.stage, args, args.device, relation_limit, out_dir)
        )
        pd.DataFrame(summaries).to_csv(out_dir / "experiment_summary.csv", index=False)

if __name__ == "__main__":
    main()