#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <embedding_source> [python_executable] [embedding_size] [gat_depth] [skip_local_preprocess:0|1] [num_factors] [kg_hops] [kg_relation_limit] [lambda_div] [lambda_sp] [split_mode]"
  exit 1
fi

EMBED_SOURCE="$1"
PYTHON_BIN="${2:-python}"
EMBEDDING_SIZE="${3:-512}"
GAT_DEPTH="${4:-1}"
SKIP_LOCAL_PREPROCESS="${5:-0}"
NUM_FACTORS="${6:-4}"
KG_HOPS="${7:-3}"
KG_RELATION_LIMIT_RAW="${8:-all}"
LAMBDA_DIV="${9:-0}"
LAMBDA_SP="${10:-0}"
SPLIT_MODE="${11:-graph}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KG_RELATION_LIMIT="$(echo "$KG_RELATION_LIMIT_RAW" | xargs)"
if [[ -z "$KG_RELATION_LIMIT" || "$KG_RELATION_LIMIT" == "0" || "$KG_RELATION_LIMIT" == "all" || "$KG_RELATION_LIMIT" == "ALL" ]]; then
  KG_RELATION_LIMIT="all"
  KG_RELATION_TAG="all"
else
  KG_RELATION_TAG="rel${KG_RELATION_LIMIT}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${EMBED_SOURCE}_emb${EMBEDDING_SIZE}_gat${GAT_DEPTH}_fac${NUM_FACTORS}_hop${KG_HOPS}_${KG_RELATION_TAG}_${TIMESTAMP}"
RUN_DIR="results/variant_runs/${RUN_ID}"
PRIMEKG_OUT="data/primekgpp_${EMBED_SOURCE}"
MODEL_OUT_DIR="models/variant_runs"

mkdir -p "$RUN_DIR"
mkdir -p "$MODEL_OUT_DIR"

if [[ "$SKIP_LOCAL_PREPROCESS" == "1" ]]; then
  echo "[1/7] Skipping local star graph preprocessing"
else
  echo "[1/7] Building local star graph artifacts"
  "$PYTHON_BIN" preprocessing/prepare_local_graph.py \
    --split-mode "$SPLIT_MODE" \
    --gnn-output data/star_graphs_gnn.pkl \
    --splits-output data/star_graph_edge_splits.json \
    --meta data/star_graphs_meta.json
fi

echo "[audit] Verifying split compatibility"
"$PYTHON_BIN" test.py

echo "[2/7] Building PrimeKG++ bundle for embedding source: ${EMBED_SOURCE}"
"$PYTHON_BIN" preprocessing/prepare_biomedkg_primekgpp.py \
  --embedding-source "$EMBED_SOURCE" \
  --output-dir "$PRIMEKG_OUT"

COMMON_TRAIN_ARGS=(
  --run-training
  --epochs 100
  --batch-size 128
  --embedding-size "$EMBEDDING_SIZE"
  --gat-depth "$GAT_DEPTH"
  --num-factors "$NUM_FACTORS"
  --learning-rate 5e-5
  --weight-decay 1e-3
  --dropout 0.4
  --label-smoothing 0.05
  --warmup-epochs 10
  --selection-metric val/loss_ce
  --early-stopping-patience 100
  --early-stopping-min-delta 0.001
)

echo "[3/7] Stage 1 training (Baseline)"
STAGE1_ARGS=(
  --stage 1
  "${COMMON_TRAIN_ARGS[@]}"
  --lambda-div "$LAMBDA_DIV"
  --lambda-sp "$LAMBDA_SP"
  --analysis-output-dir "$RUN_DIR/stage1"
)

"$PYTHON_BIN" train_star_factor_gat.py "${STAGE1_ARGS[@]}"

echo "[4/7] Stage 2 training (MoE KG-Grounded)"
STAGE2_ARGS=(
  --stage 2
  "${COMMON_TRAIN_ARGS[@]}"
  --precompute-kg-multihop
  --kg-hops "$KG_HOPS"
  --primekg-dir "$PRIMEKG_OUT"
  --lambda-div "$LAMBDA_DIV"
  --lambda-sp "$LAMBDA_SP"
  --analysis-output-dir "$RUN_DIR/stage2"
)

if [[ "$KG_RELATION_LIMIT" != "all" ]]; then
  STAGE2_ARGS+=(--kg-relation-limit "$KG_RELATION_LIMIT")
fi

"$PYTHON_BIN" train_star_factor_gat.py "${STAGE2_ARGS[@]}"

echo "[5/7] Saving model checkpoints to models/"
STAGE1_SRC="$RUN_DIR/stage1/single_run.best_model.pt"
STAGE2_SRC="$RUN_DIR/stage2/single_run.best_model.pt"
if [[ -f "$STAGE1_SRC" ]]; then cp "$STAGE1_SRC" "$MODEL_OUT_DIR/${RUN_ID}.stage1.pth"; fi
if [[ -f "$STAGE2_SRC" ]]; then cp "$STAGE2_SRC" "$MODEL_OUT_DIR/${RUN_ID}.stage2.pth"; fi

echo "[6/7] Consolidating run summaries"
export RUN_DIR
export KG_HOPS
export KG_RELATION_LIMIT
export EMBED_SOURCE
export EMBEDDING_SIZE
export GAT_DEPTH
export NUM_FACTORS
export LAMBDA_DIV
export LAMBDA_SP

"$PYTHON_BIN" - <<'PY'
import os
import pandas as pd
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
kg_relation_limit_raw = os.environ["KG_RELATION_LIMIT"]

run_metadata = {
    "run_id": run_dir.name,
    "kg_graph": os.environ["EMBED_SOURCE"],
    "embedding_size": int(os.environ["EMBEDDING_SIZE"]),
    "gat_depth": int(os.environ["GAT_DEPTH"]),
    "num_factors": int(os.environ["NUM_FACTORS"]),
    "lambda_div": float(os.environ["LAMBDA_DIV"]),
    "lambda_sp": float(os.environ["LAMBDA_SP"]),
    "kg_hops": int(os.environ["KG_HOPS"]),
    "kg_relation_limit": -1 if kg_relation_limit_raw == "all" else int(kg_relation_limit_raw),
}

rows = []
for stage_name in ["stage1", "stage2"]:
    summary_path = run_dir / stage_name / "experiment_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict.update(run_metadata)
            rows.append(row_dict)

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "run_summary.csv", index=False)
    
    all_path = Path("results/variant_runs/all_runs_summary.csv")
    if all_path.exists():
        all_df = pd.read_csv(all_path)
        all_df = pd.concat([all_df, df], ignore_index=True)
    else:
        all_df = df
    all_df.to_csv(all_path, index=False)
PY

echo "[7/7] Done: ${RUN_DIR}"