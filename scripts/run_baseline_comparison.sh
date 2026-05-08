#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python}"
SPLIT_MODE="${2:-graph}"
EPOCHS="${3:-30}"
BATCH_SIZE="${4:-64}"
DEVICE="${5:-cuda}"
OUT_ROOT="${6:-results/baseline_comparison}"
EMBEDDING_SIZE="${7:-128}"
GAT_DEPTH="${8:-1}"
KG_HOPS="${9:-2}"
KG_RELATION_LIMIT="${10:-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Rebuilding local graph artifacts with split mode: ${SPLIT_MODE}"
"$PYTHON_BIN" preprocessing/prepare_local_graph.py \
  --split-mode "$SPLIT_MODE" \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json

echo "[2/3] Auditing current splits"
"$PYTHON_BIN" test.py

mkdir -p "$OUT_ROOT"

SUMMARY_ROWS=()
WEIGHT_SUMMARIES=()

for variant in target_only context_only target_plus_context kg_context target_plus_kg target_plus_context_kg full; do
  KG_METHODS=("na")
  if [[ "$variant" == "kg_context" || "$variant" == "target_plus_kg" || "$variant" == "target_plus_context_kg" ]]; then
    KG_METHODS=("mean_decay" "gat" "dgl_gat")
  fi

  for kg_method in "${KG_METHODS[@]}"; do
    RUN_DIR="$OUT_ROOT/${variant}_${kg_method}"
    echo "[3/3] Running variant: ${variant} (kg=${kg_method})"
    STAGE=1
    EXTRA_ARGS=()
    if [[ "$variant" == "kg_context" || "$variant" == "target_plus_kg" || "$variant" == "target_plus_context_kg" ]]; then
      STAGE=2
      EXTRA_ARGS+=(--precompute-kg-multihop --kg-hops "$KG_HOPS" --kg-embed-method "$kg_method")
      if [[ "$KG_RELATION_LIMIT" != "all" ]]; then
        EXTRA_ARGS+=(--kg-relation-limit "$KG_RELATION_LIMIT")
      fi
    fi
    "$PYTHON_BIN" train_star_factor_gat.py \
      --run-training \
      --stage "$STAGE" \
      --model-variant "$variant" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --embedding-size "$EMBEDDING_SIZE" \
      --gat-depth "$GAT_DEPTH" \
      --num-factors 1 \
      --learning-rate 5e-5 \
      --weight-decay 1e-3 \
      --dropout 0.3 \
      --label-smoothing 0.05 \
      --warmup-epochs 3 \
      --analysis-output-dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}"
    SUMMARY_ROWS+=("$RUN_DIR/experiment_summary.csv")
    if [[ -f "$RUN_DIR/single_run.kg_relation_weight_summary.csv" ]]; then
      WEIGHT_SUMMARIES+=("$RUN_DIR/single_run.kg_relation_weight_summary.csv")
    fi
  done
done

echo "variant,run_name,stage,val_acc_best,test_acc,num_factors,model_variant,kg_embed_method" > "$OUT_ROOT/comparison_summary.csv"
for summary in "${SUMMARY_ROWS[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    awk -F, -v variant="$variant" 'NR==1{next} {print variant","$0}' "$summary" >> "$OUT_ROOT/comparison_summary.csv"
  fi
done

echo "variant,relation_idx,relation_id,relation_name,weight_mean,weight_std,weight_min,weight_max" > "$OUT_ROOT/kg_weight_bias_summary.csv"
for summary in "${WEIGHT_SUMMARIES[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    awk -F, -v variant="$variant" 'NR==1{next} {print variant","$0}' "$summary" >> "$OUT_ROOT/kg_weight_bias_summary.csv"
  fi
done

echo "Baseline comparison complete: ${OUT_ROOT}"
echo "Summary CSV: ${OUT_ROOT}/comparison_summary.csv"
echo "KG bias summary CSV: ${OUT_ROOT}/kg_weight_bias_summary.csv"