# scGNN-Factors: Technical Spec + Proposal Alignment Review

This README is the reviewable technical spec decomposition of the proposal in [spec_proposal.tex](spec_proposal.tex), with a direct alignment audit of current implementation status.

## Scope

- Problem: 188-way edge-class prediction on local star graphs using fixed scGPT node embeddings.
- Two experiments:
  - Stage 1: local data-driven edge modulation.
  - Stage 2: PrimeKG++ conditioned edge modulation.
- Hard constraints:
  - no target-label leakage in feature construction,
  - full-candidate scoring over all 188 classes.

## Implemented File Map

- [train_star_factor_gat.py](train_star_factor_gat.py)
  - Full training loop, KG ablation sweeps, edge-dropout regularization, early stopping, and analysis export.

If a `.env` file exists in the repo root and contains `WANDB_API_KEY`, training will automatically log metrics to the W&B project `factor-kg`.
- [pipeline/models/edge_hypernetwork.py](pipeline/models/edge_hypernetwork.py)
  - Basis tensor, local/KG routing weights, candidate edge construction.
- [pipeline/models/stage1_baseline.py](pipeline/models/stage1_baseline.py)

  - Stage 2 full-candidate scoring with multirelation KG pooling and KG-conditioned routing.
- [pipeline/primekg_bridge.py](pipeline/primekg_bridge.py)
  - Class mapping, relation-aware PrimeKG adjacency loading, base KG tensor build, multirelation multihop precompute/cache.
- [pipeline/star_data.py](pipeline/star_data.py)
  - Training sample construction from star-graph artifacts.
- [pipeline/losses.py](pipeline/losses.py)
  - CE + independence + entropy sparsity objective.
- [pipeline/train_eval.py](pipeline/train_eval.py)
  - Train/eval steps with leakage-structure assertions.

## Data Contracts Used

- Local graph artifacts:
  - [data/star_graphs_gnn.pkl](data/star_graphs_gnn.pkl)
  - [data/star_graph_edge_splits.json](data/star_graph_edge_splits.json)
  - [data/star_graphs_meta.json](data/star_graphs_meta.json)
- PrimeKG++ artifacts:

If a `.env` file exists in the repo root and contains `WANDB_API_KEY`, training will automatically log metrics to the W&B project `factor-kg`.
  - [data/primekgpp_grace_redaf/node_embeddings.npy](data/primekgpp_grace_redaf/node_embeddings.npy)
  - [data/primekgpp_grace_redaf/entity2id.txt](data/primekgpp_grace_redaf/entity2id.txt)
  - [data/primekgpp_grace_redaf/kg2id.txt](data/primekgpp_grace_redaf/kg2id.txt)
- Mapping:
  - [tahoe_to_primekg_map.csv](tahoe_to_primekg_map.csv)

## Alignment Matrix vs Proposal

### Fully Aligned

- Modular code decomposition matches planned modules from proposal.
- Stage 1 and Stage 2 separation is present.
- Hypernetwork structure (`W_k`, `z_c`, `A_local`, `A_KG`) matches formulation.

## Current Answer to “Are they all aligned?”


When `WANDB_API_KEY` is present in `.env`, the W&B project `factor-kg` also logs:

- train / val / test losses,
- all tracked evaluation metrics,
- per-epoch Stage 2 relation-weight trajectories,
- per-epoch mechanism summaries,
- final run summary metrics.

1. Build relation-aware adjacency:
  - `adj[head_id][relation_id] = {tail_id_1, tail_id_2, ...}`
2. For each of the 188 classes, collect multi-hop neighbors grouped by relation.
3. Mean pool node embeddings within each relation bucket.
4. Assemble a class-aligned tensor $H_{KG}^{multi} \in \mathbb{R}^{188 \times R \times 256}$.
5. In [pipeline/models/stage2_primekg.py](pipeline/models/stage2_primekg.py), use the learnable class bottleneck to predict relation weights per class.
6. Compute a class-conditioned weighted average over relations to obtain the final KG signature used by the edge hypernetwork.

This means Stage 2 no longer collapses all KG neighbors into one global mean before training. The model now learns how much each relation type should contribute for each class.

## Explicit 3-Hop PrimeKG++ Precompute

- Enabled via [train_star_factor_gat.py](train_star_factor_gat.py):
  - `--precompute-kg-multihop --kg-hops 3`
- Backend function:
  - `precompute_h_kg_multihop(..., hops=3)` in [pipeline/primekg_bridge.py](pipeline/primekg_bridge.py)
- Cache payload:
  - `h_kg_multi`: relation-bucketed tensor `[C, R, d_kg]`
  - `relation_ids`: retained PrimeKG relation ids aligned to axis `R`

## Data Preparation Status

- Implemented sample builder/collation in [pipeline/star_data.py](pipeline/star_data.py).
- Produces model-ready tensors:
  - `h0`, `h_known`, `y_known`, `h_target`, `y`
- Training now uses train-only random edge masking: each epoch re-samples one held-out positive edge per train star occurrence, while validation/test keep fixed held-out edges for stable benchmarking.
- Stage 2 relation aggregation weights are tracked per epoch in the history CSV and can be streamed to W&B.

## How to Run

### 1) Environment

From repo root:

```bash
cd /home/jethrocsau/scGNN-Factors
```

Use your project Python:

```bash
python --version
```

If you use a named environment such as the scGPT environment, activate it first and then run the commands below with `python`.

If a `.env` file exists in the repo root and contains `WANDB_API_KEY`, training will automatically log metrics to the W&B project `factor-kg`.

### 1.5) If embeddings are already done (recommended for your current state)

If you already embedded a subset (for example ~7M records) and have
`data/tahoe_embeddings_parquet*.npz`, start from here:

1. Build local star graphs + train/val/test splits once.
2. Sanity-check data loading.
3. Run one variant first.
4. Launch the full sweep.

Commands:

```bash
# 1) Build local graph artifacts + splits from existing embedding NPZ files
python preprocessing/prepare_local_graph.py \
  --input data/tahoe_embeddings_parquet*.npz \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json

# 2) Sanity-check that train split tensors load
python train_star_factor_gat.py \
  --prepare-data --split train --batch-size 16 --stage 1

# 3) Run one variant first (fast validation before sweep; defaults to --ind-mode cosine and --warmup-epochs 10)
bash scripts/run_variant_pipeline.sh grace_redaf python 512 1 1

# 4) Run full sweep (grace/ggd/dgi mapped to *_redaf)
bash scripts/run_primekg_embedding_sweep.sh python grace,ggd,dgi 512,384 1,2
```

Notes:

- The final `1` in `run_variant_pipeline.sh ... 1` skips local preprocessing, because you already created splits in step 1.
- `run_variant_pipeline.sh` now defaults to `--ind-mode cosine` for a cheaper independence penalty. To override it, pass a sixth argument such as `distance`.
- `run_variant_pipeline.sh` also defaults to `--warmup-epochs 10`, which disables the auxiliary regularizers for the first 10 epochs and then turns them on.
- Stage checkpoints are copied to `models/variant_runs/*.pth` and metrics go to `results/variant_runs/`.

### 2) Preprocess local embeddings (if needed)

If your `data/tahoe_embeddings_parquet*.npz` shards do not exist yet, run embedding prep first:

```bash
python preprocessing/prepare_embeds.py \
  --start-index 0 --parquet-shard-size 400 --shard-index 0
```

### 3) Build/refresh local star-graph artifacts + train/val/test splits (required before training)

```bash
python preprocessing/prepare_local_graph.py \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json
```

### 4) Build/refresh PrimeKG++ bundle (if needed)

```bash
python preprocessing/prepare_biomedkg_primekgpp.py --output-dir data/primekgpp_grace_redaf
```

### 5) Data preparation preview for training

This confirms tensors are created correctly from split artifacts:

```bash
python train_star_factor_gat.py \
  --prepare-data --split train --batch-size 16 --stage 1
```

### 6) Stage 1 (local-only) model init path

```bash
python train_star_factor_gat.py --stage 1
```

### 7) Stage 2 (PrimeKG++) model init path

Use base class-mapped KG tensor:

```bash
python train_star_factor_gat.py --stage 2
```

Use precomputed multi-hop KG tensor (default request: 3 hops):

```bash
python train_star_factor_gat.py \
  --stage 2 --precompute-kg-multihop --kg-hops 3
```

Force recompute of cached multihop tensor:

```bash
python train_star_factor_gat.py \
  --stage 2 --precompute-kg-multihop --kg-hops 3 --force-kg-recompute
```

### 8) Current limitation

`train_star_factor_gat.py` now supports full epoch-based training and exports analysis artifacts. The scaffold/init path is still available when `--run-training` is not used.

### 9) Full training loop (single run)

Stage 1 (no KG baseline):

```bash
python train_star_factor_gat.py \
  --stage 1 --run-training --epochs 20 --batch-size 32 \
  --embedding-size 512 --gat-depth 1 \
  --warmup-epochs 10 \
  --selection-metric val/loss_ce --early-stopping-patience 5 --early-stopping-min-delta 0.001
```

Stage 2 (KG multihop):

```bash
python train_star_factor_gat.py \
  --stage 2 --precompute-kg-multihop --kg-hops 3 --run-training --epochs 20 --batch-size 32 \
  --embedding-size 512 --gat-depth 1 \
  --warmup-epochs 10 \
  --selection-metric val/loss_ce --early-stopping-patience 5 --edge-dropout-rate 0.1
```

Resume an interrupted run from checkpoint:

```bash
python train_star_factor_gat.py \
  --stage 2 --precompute-kg-multihop --kg-hops 3 --run-training --epochs 20 --batch-size 32 \
  --embedding-size 512 --gat-depth 1 \
  --warmup-epochs 10 \
  --resume-run --checkpoint-every 1 --analysis-output-dir results/star_factor_analysis
```

Training uses `tqdm` progress bars with live running train losses and per-epoch validation summaries.

### 10) KG-size ablation for factor analysis

Runs baseline (no KG) + Stage 2 with increasing KG relation limits:

```bash
python train_star_factor_gat.py \
  --run-training --run-kg-ablation --precompute-kg-multihop --kg-hops 3 \
  --kg-relation-limits 1,2,4,8,16 --epochs 20 --batch-size 32 \
  --selection-metric val/f1_macro --early-stopping-patience 5
```

Outputs are written to [results/star_factor_analysis](results/star_factor_analysis):

- `experiment_summary.csv` (val/test metrics + factor summary for regression analysis)
- `*.history.csv` (per-epoch metrics, including Stage 2 relation-weight trajectories)
- `*.factor_weights.csv` (mechanism weight summaries)
- `*.factor_weights_by_drug.csv` (class/drug-level factor weights)
- `*.relation_weights.csv` (global KG relation-weight summaries)
- `*.relation_weights_by_drug.csv` (class/drug-level KG relation weights)
- `*.val_predicted_drug_metrics.csv`, `*.test_predicted_drug_metrics.csv` (predicted-drug analysis)
- `*.val.prediction_details.csv`, `*.test.prediction_details.csv` (sample-level predictions with confidence, rank, top-k classes, predicted mechanism weights, and predicted KG relation weights)
- `*.val.relation_prediction_summary.csv`, `*.test.relation_prediction_summary.csv` (dominant KG relation usage and accuracy by predicted class)
- `*.summary.json` (run metadata and best-epoch selection metric)
- `*.loss_curves.png`, `*.accuracy_curves.png`, `*.backprop_curves.png`
- `*.val.confusion_matrix.csv`, `*.test.confusion_matrix.csv`
- `*.val.confusion_matrix.png`, `*.test.confusion_matrix.png`
- `*.val.calibration_bins.csv`, `*.test.calibration_bins.csv`
- `*.val.reliability_curve.png`, `*.test.reliability_curve.png`
- `*.checkpoint.pt` (periodic training checkpoint; resumable)
- `*.best_model.pt`

The summary and history files now also include enhanced evaluation metrics:

- balanced accuracy,
- mean reciprocal rank (MRR),
- mean true-class probability,
- Brier score,
- expected calibration error (ECE).

When `WANDB_API_KEY` is present in `.env`, the W&B project `factor-kg` also logs:

- train / val / test losses,
- all tracked evaluation metrics,
- per-epoch Stage 2 relation-weight trajectories,
- per-epoch mechanism summaries,
- final run summary metrics.

### 11) Variant automation scripts

Recommended sequence with scripts:

1. Preprocess local graph artifacts once.
2. Run variant sweeps (PrimeKG bundle per embedding key + Stage1/2 training).

Run one embedding variant end-to-end:

```bash
bash scripts/run_variant_pipeline.sh grace_redaf python 512 1 1
```

To force the original distance-correlation penalty instead of cosine:

```bash
bash scripts/run_variant_pipeline.sh grace_redaf python 512 1 1 distance
```

The final `1` skips local preprocessing because step 1 already generated splits.

Run the default sweep (`grace_redaf`, `ggd_redaf`, `dgi_redaf`) after one-time local preprocessing:

```bash
bash scripts/run_primekg_embedding_sweep.sh python grace,ggd,dgi 512,384 1,2
```

Outputs go under [results/variant_runs](results/variant_runs) with aggregate summary at [results/variant_runs/all_runs_summary.csv](results/variant_runs/all_runs_summary.csv).

Additional cross-stage factor comparison (per sweep run):

- [results/variant_runs/<RUN_ID>/alpha_local_vs_kg_by_drug.csv](results/variant_runs) contains per-drug comparison between Stage 1 and Stage 2 routing distributions (`alpha_local` vs `alpha_kg`) with:
  - cosine similarity,
  - Jensen-Shannon divergence (JSD).
- `run_summary.csv` and `all_runs_summary.csv` now include aggregate stats:
  - `alpha_compare/cosine_*` and `alpha_compare/jsd_*`.

## Recommended Next Steps

1. Extend relation/mechanism diagnostics with cross-run comparisons across PrimeKG variants.
2. Add stricter leakage audit logs (feature-construction path checks beyond logits shape).
3. Optionally implement a richer within-relation aggregator beyond mean pooling.
