#!/usr/bin/env bash
set -euo pipefail

# Run a single StarFactor/baseline variant with user-set hyperparameters.
#
# Examples:
#   bash scripts/run_single_variant_scgpt_final.sh
#   VARIANT=target_plus_kg KG_SOURCE_KEY=dgi KG_METHOD=mean_decay KG_HOPS=2 KG_RELATION_LIMIT=2 bash scripts/run_single_variant_scgpt_final.sh
#   VARIANT=target_plus_context LEARNING_RATE=5e-5 DROPOUT=0.2 bash scripts/run_single_variant_scgpt_final.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_MODE="${SPLIT_MODE:-graph}"
LOCAL_EMBED_INPUT="${LOCAL_EMBED_INPUT:-data/scgpt_embeds/tahoe_embeddings_parquet*.npz}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"

# -----------------------------
# User-set run configuration
# -----------------------------
VARIANT="${VARIANT:-target_plus_context_kg}"
KG_SOURCE_KEY_RAW="${KG_SOURCE_KEY:-grace}"
KG_METHOD="${KG_METHOD:-path_attn}"
KG_HOPS="${KG_HOPS:-2}"
KG_RELATION_LIMIT="${KG_RELATION_LIMIT:-all}"   # use 'all' for no explicit cap
KG_PATH_MAX_PATHS="${KG_PATH_MAX_PATHS:-64}"

EPOCHS="${EPOCHS:-500}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
SELECTION_METRIC="${SELECTION_METRIC:-val/loss_ce}"

BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
EMBEDDING_SIZE="${EMBEDDING_SIZE:-64}"
INPUT_EMBED_DIM="${INPUT_EMBED_DIM:-512}"
GAT_DEPTH="${GAT_DEPTH:-1}"
GAT_HIDDEN_DIM="${GAT_HIDDEN_DIM:-168}"
NUM_FACTORS="${NUM_FACTORS:-1}"
DROPOUT="${DROPOUT:-0.3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
LAMBDA_DIV="${LAMBDA_DIV:-1e-4}"
LAMBDA_SP="${LAMBDA_SP:-1e-4}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
SEED="${SEED:-42}"

OUT_ROOT="${OUT_ROOT:-results/single_variant_runs}"
RUN_TAG="${RUN_TAG:-}"

normalize_embedding_key() {
  local key
  key="$(echo "${1:-}" | xargs)"
  case "$key" in
    grace|ggd|dgi)
      printf '%s_redaf\n' "$key"
      ;;
    primekgpp_*)
      printf '%s\n' "${key#primekgpp_}"
      ;;
    *)
      printf '%s\n' "$key"
      ;;
  esac
}

resolve_primekg_dir() {
  local key="$1"
  printf '%s/data/primekgpp_%s\n' "$ROOT_DIR" "$key"
}

is_non_kg_variant() {
  case "$1" in
    target_only|context_only|target_plus_context)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

KG_SOURCE_KEY="$(normalize_embedding_key "$KG_SOURCE_KEY_RAW")"
PRIMEKG_DIR="$(resolve_primekg_dir "$KG_SOURCE_KEY")"

if [[ ! -d "$PRIMEKG_DIR" ]]; then
  echo "Missing PrimeKG directory: $PRIMEKG_DIR" >&2
  exit 1
fi

if is_non_kg_variant "$VARIANT"; then
  STAGE=1
  KG_METHOD="na"
  KG_HOPS="na"
  KG_RELATION_LIMIT="all"
else
  STAGE=2
fi

if [[ "$OUT_ROOT" = /* ]]; then
  OUT_ROOT_ABS="$OUT_ROOT"
else
  OUT_ROOT_ABS="$ROOT_DIR/$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT_ABS/$KG_SOURCE_KEY"

REL_TAG="$KG_RELATION_LIMIT"
HOP_TAG="$KG_HOPS"
METHOD_TAG="$KG_METHOD"
RUN_DIR_NAME="${VARIANT}_${METHOD_TAG}_hop${HOP_TAG}_rel${REL_TAG}"
if [[ -n "$RUN_TAG" ]]; then
  RUN_DIR_NAME="${RUN_DIR_NAME}_${RUN_TAG}"
fi
RUN_DIR="$OUT_ROOT_ABS/$KG_SOURCE_KEY/$RUN_DIR_NAME"
mkdir -p "$RUN_DIR"

echo "========================================"
echo "Single-variant training run"
echo "ROOT_DIR               : $ROOT_DIR"
echo "VARIANT                : $VARIANT"
echo "STAGE                  : $STAGE"
echo "KG_SOURCE_KEY          : $KG_SOURCE_KEY"
echo "KG_METHOD              : $KG_METHOD"
echo "KG_HOPS                : $KG_HOPS"
echo "KG_RELATION_LIMIT      : $KG_RELATION_LIMIT"
echo "EPOCHS                 : $EPOCHS"
echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
echo "RUN_DIR                : $RUN_DIR"
echo "========================================"

if [[ "$SKIP_PREP" != "1" ]]; then
  echo "[1/3] Rebuilding local graph artifacts with split mode: ${SPLIT_MODE}"
  "$PYTHON_BIN" preprocessing/prepare_local_graph.py \
    --input "$LOCAL_EMBED_INPUT" \
    --split-mode "$SPLIT_MODE" \
    --gnn-output data/star_graphs_gnn.pkl \
    --splits-output data/star_graph_edge_splits.json \
    --meta data/star_graphs_meta.json
else
  echo "[1/3] Skipping local graph rebuild"
fi

if [[ "$SKIP_AUDIT" != "1" ]]; then
  echo "[2/3] Auditing current splits"
  "$PYTHON_BIN" test.py
else
  echo "[2/3] Skipping split audit"
fi

CMD=(
  "$PYTHON_BIN" train_star_factor_gat.py
  --run-training
  --stage "$STAGE"
  --model-variant "$VARIANT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --input-embed-dim "$INPUT_EMBED_DIM"
  --embedding-size "$EMBEDDING_SIZE"
  --gat-depth "$GAT_DEPTH"
  --gat-hidden-dim "$GAT_HIDDEN_DIM"
  --num-factors "$NUM_FACTORS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --lambda-div "$LAMBDA_DIV"
  --lambda-sp "$LAMBDA_SP"
  --dropout "$DROPOUT"
  --label-smoothing "$LABEL_SMOOTHING"
  --warmup-epochs "$WARMUP_EPOCHS"
  --selection-metric "$SELECTION_METRIC"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  --analysis-output-dir "$RUN_DIR"
  --primekg-dir "$PRIMEKG_DIR"
  --seed "$SEED"
)

if [[ "$STAGE" -eq 2 ]]; then
  CMD+=(
    --precompute-kg-multihop
    --kg-hops "$KG_HOPS"
    --kg-embed-method "$KG_METHOD"
    --kg-path-max-paths "$KG_PATH_MAX_PATHS"
  )
  if [[ "$KG_RELATION_LIMIT" != "all" ]]; then
    CMD+=(--kg-relation-limit "$KG_RELATION_LIMIT")
  fi
fi

echo "[3/3] Launching training"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo
echo "Run complete. Outputs written to: $RUN_DIR"
echo "Key files:"
echo " - $RUN_DIR/experiment_summary.csv"
echo " - $RUN_DIR/single_run.run_config.json"
echo " - $RUN_DIR/single_run_history.csv"
echo " - $RUN_DIR/single_run.test_predictions.csv"
echo " - $RUN_DIR/single_run.kg_relation_weight_summary.csv"
echo " - $RUN_DIR/single_run.kg_path_weight_summary.csv"
