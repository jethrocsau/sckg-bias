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

try:
    import wandb
except ImportError:
    wandb = None

from pipeline.models.baselines import ContextOnlyBaseline
from pipeline.models.baselines import KGContextBaseline
from pipeline.models.baselines import TargetPlusContextBaseline
from pipeline.models.baselines import TargetPlusContextKGBaseline
from pipeline.models.baselines import TargetPlusKGBaseline
from pipeline.models.baselines import TargetOnlyBaseline
from pipeline.models.star_factor import StarFactorModel
from pipeline.primekg_bridge import PrimeKGBridgeConfig
from pipeline.primekg_bridge import build_h_kg_class_tensor
from pipeline.primekg_bridge import load_drug_class_order
from pipeline.primekg_bridge import load_multirel_relation_ids
from pipeline.primekg_bridge import load_relation_id_to_name
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
        choices=["mean_decay", "gat", "dgl_gat"],
        help="How KG relation embeddings are pooled for KG-aware baselines: uniform mean or relation-GAT. `dgl_gat` is kept as a backward-compatible alias for `gat`.",
    )
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
        choices=["full", "target_only", "context_only", "kg_context", "target_plus_context", "target_plus_kg", "target_plus_context_kg"],
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
        default=0, 
        help="Number of epochs to wait for improvement before stopping."
    )
    parser.add_argument(
        "--early-stopping-min-delta", 
        type=float, 
        default=0.0, 
        help="Minimum change to qualify as an improvement."
    )
    
    return parser.parse_args()

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
    )
    if args.model_variant == "target_only":
        return TargetOnlyBaseline(cfg).to(device)
    if args.model_variant == "context_only":
        return ContextOnlyBaseline(cfg).to(device)
    if args.model_variant == "kg_context":
        return KGContextBaseline(cfg).to(device)
    if args.model_variant == "target_plus_context":
        return TargetPlusContextBaseline(cfg).to(device)
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


def _export_relation_weight_biases(model, h_kg_multi, relation_ids, relation_names, class_names, out_dir, run_name):
    if not hasattr(model, "export_relation_weights") or h_kg_multi is None:
        return
    relation_weights = model.export_relation_weights(h_kg_multi)
    if relation_weights is None:
        return

    names = relation_names or []
    ids = relation_ids or []
    if relation_weights.shape[1] > len(names):
        names = ["self", *names]
        ids = [-1, *ids]

    rows = []
    for class_idx in range(relation_weights.shape[0]):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        for rel_idx in range(relation_weights.shape[1]):
            rel_name = names[rel_idx] if rel_idx < len(names) else f"relation_{rel_idx}"
            rel_id = ids[rel_idx] if rel_idx < len(ids) else rel_idx
            rows.append(
                {
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "relation_idx": rel_idx,
                    "relation_id": rel_id,
                    "relation_name": rel_name,
                    "weight": float(relation_weights[class_idx, rel_idx]),
                }
            )

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

def _run_single_training(run_name, stage, args, device, relation_limit, out_dir):
    _validate_closed_set_split(StarDataConfig())

    class_names = load_drug_class_order(meta_path=args.star_meta_path)
    bridge_cfg = PrimeKGBridgeConfig(primekg_dir=args.primekg_dir)

    relation_ids, relation_names = None, None
    if stage == 1 or args.model_variant in {"target_only", "context_only", "target_plus_context"}:
        h_kg_multi = None
        num_kg_relations = 1
        d_kg = 256
    else:
        h_kg_multi = precompute_h_kg_multihop(bridge_cfg, class_names, hops=args.kg_hops, relation_limit=relation_limit).to(device)
        num_kg_relations = h_kg_multi.shape[1]
        d_kg = h_kg_multi.shape[2]
        
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

    history_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = -1.0
    best_model_state = None
    
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
            "val/acc": val_eval["acc"]
        }
        history_rows.append(row)
        
        # Update Main Epoch Bar with current scores
        epoch_pbar.set_postfix({
            "tr_ce": f"{row['train/loss_ce']:.3f}", 
            "val_ce": f"{row['val/loss_ce']:.3f}", 
            "val_acc": f"{row['val/acc']:.3f}"
        })
        
        # Checkpointing
        if val_eval["acc"] > best_val_acc:
            best_val_acc = val_eval["acc"]
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # --- Post-Training Interpretability Generation ---
    print(f"[{run_name}] Training complete. Best Val Acc: {best_val_acc:.4f}. Generating meta-analysis artifacts...")
    
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
    
    # 3. Factor-Relation Map Export
    if stage == 2:
        _export_factor_analysis(model, relation_ids, relation_names, class_names, out_dir, run_name)
    _export_relation_weight_biases(model, h_kg_multi, relation_ids, relation_names, class_names, out_dir, run_name)
    
    return {
        "run_name": run_name,
        "stage": stage,
        "val_acc_best": float(best_val_acc),
        "test_acc": float(test_details["acc"]),
        "num_factors": args.num_factors,
        "model_variant": args.model_variant,
        "kg_embed_method": args.kg_embed_method,
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