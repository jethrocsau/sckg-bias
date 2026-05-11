# scKG-Bias

Closed-set star-graph edge-class prediction for single-cell perturbation data, with optional PrimeKG++ context and interpretability exports.

## Overview

This repository implements the training and analysis pipeline used for `scKG-Bias` experiments:

- local star-graph construction from embedding shards,
- optional PrimeKG++ preprocessing from BioMedKG assets,
- training of full and ablation baselines,
- export of relation/path attribution summaries.

The target task is closed-set edge-class prediction over 188 classes.

## Final repository entrypoints

- `train_star_factor_gat.py` — main trainer and export entrypoint.
- `pipeline/` — models, losses, KG bridge, dataloading, train/eval utilities.
- `preprocessing/prepare_local_graph.py` — builds local graph payload + splits.
- `preprocessing/prepare_biomedkg_primekgpp.py` — builds PrimeKG++-style artifacts.
- `preprocessing/prepare_embeds_scgpt.py` — scGPT embedding shard generation.
- `test.py` — split-overlap and leakage audit for generated local graph splits.
- `visualize_embed.py` — embedding-space visualization helper.
- `visualize_primekgpp.py` — PrimeKG++ visualization helper.
- `working/analysis.ipynb` — analysis notebook for publication figures and artifact audits.
- `scripts/run_baseline_comparison_scgpt_all.sh` — multi-source baseline/KG ablation sweep.
- `scripts/run_baseline_comparison_scgpt_grace_max2hop.sh` — sweep script for `grace_redaf` with max 2 hops.
- `scripts/run_baseline_comparison_scgpt_ggd_max2hop.sh` — sweep script for `ggd_redaf` with max 2 hops.
- `scripts/run_baseline_comparison_scgpt_dgi_max2hop.sh` — sweep script for `dgi_redaf` with max 2 hops.
- `scripts/run_single_variant_scgpt_final.sh` — single-run launcher with user-set hyperparameters.

## Required artifacts

### Local graph artifacts (training input)

- `data/star_graphs_gnn.pkl`
- `data/star_graph_edge_splits.json`
- `data/star_graphs_meta.json`

### KG bundle (required for KG-enabled variants)

Example: `data/primekgpp_grace_redaf/` with at least:

- `node_embeddings.npy`
- `entity2id.txt`
- `kg2id.txt`
- `relation2id.txt`

### Mapping file

- `tahoe_to_primekg_map.csv`

## Quick start

### 1) Build local star-graph artifacts

```bash
python preprocessing/prepare_local_graph.py \
  --input "data/scgpt_embeds/tahoe_embeddings_parquet*.npz" \
  --split-mode graph \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json
```

### 2) (Optional) Build PrimeKG++ bundle

```bash
python preprocessing/prepare_biomedkg_primekgpp.py \
  --kg-data-dir kg/data \
  --embedding-source grace_redaf \
  --output-dir data/primekgpp_grace_redaf
```

### 3) (Optional) Build scGPT embedding shards

```bash
python preprocessing/prepare_embeds_scgpt.py \
  --parquet-shard-size 400 \
  --shard-index 0
```

## Training

`train_star_factor_gat.py` runs training only when `--run-training` is set.

### Stage 1 (local-only)

```bash
python train_star_factor_gat.py \
  --run-training \
  --stage 1 \
  --model-variant target_plus_context \
  --epochs 20 \
  --batch-size 16 \
  --device cuda \
  --analysis-output-dir results/star_factor_analysis
```

### Stage 2 (KG-enabled)

```bash
python train_star_factor_gat.py \
  --run-training \
  --stage 2 \
  --model-variant target_plus_context_kg \
  --primekg-dir data/primekgpp_grace_redaf \
  --precompute-kg-multihop \
  --kg-hops 3 \
  --kg-embed-method gat \
  --kg-subtree-pool-method match \
  --kg-relation-pool-method match \
  --kg-relation-limit 0 \
  --epochs 20 \
  --batch-size 16 \
  --device cuda \
  --analysis-output-dir results/star_factor_analysis
```

### Early stopping knobs

- `--selection-metric` (default `val/loss_ce`)
- `--early-stopping-patience` (default `10`; `0` disables)
- `--early-stopping-min-delta` (default `0.001`)

## Single-variant runner

For one-off experiments, use the single-run launcher:

```bash
bash scripts/run_single_variant_scgpt_final.sh
```

Default behavior:

- trains exactly one variant,
- runs for up to `500` epochs,
- uses early stopping with patience `30`,
- rebuilds local graph artifacts and audits splits unless disabled.

Common overrides are passed as environment variables:

```bash
VARIANT=target_plus_kg \
KG_SOURCE_KEY=dgi \
KG_METHOD=mean_decay \
KG_HOPS=2 \
KG_RELATION_LIMIT=2 \
LEARNING_RATE=5e-5 \
DROPOUT=0.2 \
bash scripts/run_single_variant_scgpt_final.sh
```

Useful variables:

- `VARIANT`
- `KG_SOURCE_KEY`
- `KG_METHOD`
- `KG_HOPS`
- `KG_RELATION_LIMIT` (`all` leaves the cap unset)
- `EPOCHS`
- `EARLY_STOPPING_PATIENCE`
- `EARLY_STOPPING_MIN_DELTA`
- `LEARNING_RATE`
- `DROPOUT`
- `BATCH_SIZE`
- `OUT_ROOT`
- `RUN_TAG`
- `SKIP_PREP=1`
- `SKIP_AUDIT=1`

Stage selection is automatic:

- `target_only`, `context_only`, `target_plus_context` run as stage `1`
- KG-enabled variants run as stage `2`

## Baseline + KG sweep

Run matrix sweeps with:

```bash
bash scripts/run_baseline_comparison_scgpt_all.sh \
  python graph 30 16 cuda results/baseline_comparison 64 1 \
  1,2 1,2,3,4 grace mean_decay,path_attn \
  "data/scgpt_embeds/tahoe_embeddings_parquet*.npz"
```

Current sweep launchers in `scripts/`:

- `run_baseline_comparison_scgpt_all.sh` — runs across `dgi`, `ggd`, and `grace`
- `run_baseline_comparison_scgpt_grace_max2hop.sh` — `grace_redaf` only
- `run_baseline_comparison_scgpt_ggd_max2hop.sh` — `ggd_redaf` only
- `run_baseline_comparison_scgpt_dgi_max2hop.sh` — `dgi_redaf` only

Script behavior:

1. Rebuilds local graph artifacts.
2. Audits split integrity via `test.py`.
3. Trains baseline variants across configured KG settings.
4. Writes merged summary CSVs under the selected output root.

## Main outputs

Per run directory (e.g. `results/.../single_run*`):

- `single_run.run_config.json`
- `single_run_history.csv`
- `single_run.best_model.pt`
- `single_run.val_predictions.csv`
- `single_run.test_predictions.csv`
- `single_run.prediction_summary.csv`
- `single_run.prediction_comparison.csv`
- `single_run.factor_kg_mappings.csv` (stage-2 factor export when available)
- `single_run.kg_relation_weights.csv`
- `single_run.kg_relation_weight_summary.csv`
- `single_run.kg_path_weights.csv` (path-attn mode)
- `single_run.kg_path_weight_summary.csv` (path-attn mode)
- `single_run.kg_subtree_members.csv` (when subtree-member cache is available)
- `experiment_summary.csv`

Sweep-level merged outputs:

- `comparison_summary.csv`
- `kg_weight_bias_summary.csv`
- `kg_path_bias_summary.csv`
- `prediction_summary.csv`
- `prediction_comparison.csv`

Notebook-generated analysis tables and figures are written under:

- `working/figs_pub/`
- `working/figs_pub/tables/`

## Notes for reproducibility

- Training is closed-set; the trainer raises if val/test contain unseen classes relative to train.
- KG context for held-out target edges is constructed from known neighbors, not direct target indexing in the local task pipeline.
- If relation/path member exports are sparse, recompute KG caches with matching hop/relation settings.
- Shared non-KG baseline runs may appear under `results/baseline_comparison/_shared/` and be symlinked into source-specific folders.

## Optional logging

If `wandb` is installed, runs can be integrated with Weights & Biases (depends on local configuration).

## Project status

`experiments/` is retained as legacy material and is not required for the current training/sweep workflow.
