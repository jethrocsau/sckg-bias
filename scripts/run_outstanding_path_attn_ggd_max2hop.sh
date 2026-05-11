#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python}"
SPLIT_MODE="${2:-graph}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-16}"
DEVICE="${5:-cuda}"
OUT_ROOT="${6:-results/baseline_comparison}"
EMBEDDING_SIZE="${7:-64}"
GAT_DEPTH="${8:-1}"
KG_RELATION_LIMIT="${9:-1,2,4,6,8,all}"
LOCAL_EMBED_INPUT="${10:-data/scgpt_embeds/tahoe_embeddings_parquet*.npz}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_KEY="ggd_redaf"
KG_VARIANTS=(context_plus_kg target_plus_kg target_plus_context_kg)
KG_HOPS_LIST=(1 2)
IFS=',' read -r -a KG_RELATION_LIMIT_LIST <<< "$KG_RELATION_LIMIT"

if [[ "$OUT_ROOT" = /* ]]; then
  OUT_ROOT_ABS="$OUT_ROOT"
else
  OUT_ROOT_ABS="$ROOT_DIR/$OUT_ROOT"
fi

resolve_primekg_dir() {
  local key="$1"
  printf '%s/data/primekgpp_%s\n' "$ROOT_DIR" "$key"
}

PRIMEKG_DIR="$(resolve_primekg_dir "$SOURCE_KEY")"
if [[ ! -d "$PRIMEKG_DIR" ]]; then
  echo "Missing PrimeKG directory: $PRIMEKG_DIR" >&2
  exit 1
fi

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "[1/3] Rebuilding local graph artifacts with split mode: ${SPLIT_MODE}"
  "$PYTHON_BIN" preprocessing/prepare_local_graph.py \
    --input "$LOCAL_EMBED_INPUT" \
    --split-mode "$SPLIT_MODE" \
    --gnn-output data/star_graphs_gnn.pkl \
    --splits-output data/star_graph_edge_splits.json \
    --meta data/star_graphs_meta.json

  echo "[2/3] Auditing current splits"
  "$PYTHON_BIN" test.py
else
  echo "[1/3] SKIP_PREP=1, reusing existing local graph artifacts"
fi

mkdir -p "$OUT_ROOT_ABS/$SOURCE_KEY"

echo "[3/3] Running outstanding path_attn max-2-hop ablations for $SOURCE_KEY"
for variant in "${KG_VARIANTS[@]}"; do
  for kg_hops in "${KG_HOPS_LIST[@]}"; do
    for kg_relation_limit in "${KG_RELATION_LIMIT_LIST[@]}"; do
      kg_relation_limit="$(echo "$kg_relation_limit" | xargs)"
      [[ -z "$kg_relation_limit" ]] && kg_relation_limit="all"

      RUN_DIR="$OUT_ROOT_ABS/${SOURCE_KEY}/${variant}_path_attn_hop${kg_hops}_rel${kg_relation_limit}"
      if [[ -f "$RUN_DIR/experiment_summary.csv" ]]; then
        echo "[skip] already exists: $RUN_DIR"
        continue
      fi

      EXTRA_ARGS=(
        --primekg-dir "$PRIMEKG_DIR"
        --precompute-kg-multihop
        --kg-hops "$kg_hops"
        --kg-embed-method path_attn
      )
      if [[ "$kg_relation_limit" != "all" ]]; then
        EXTRA_ARGS+=(--kg-relation-limit "$kg_relation_limit")
      fi

      echo "[run] source=${SOURCE_KEY} variant=${variant} kg=path_attn hops=${kg_hops} rel_limit=${kg_relation_limit}"
      "$PYTHON_BIN" train_star_factor_gat.py \
        --run-training \
        --stage 2 \
        --model-variant "$variant" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --embedding-size "$EMBEDDING_SIZE" \
        --gat-depth "$GAT_DEPTH" \
        --num-factors 1 \
        --learning-rate 1e-4 \
        --weight-decay 1e-3 \
        --dropout 0.3 \
        --label-smoothing 0.05 \
        --warmup-epochs 3 \
        --analysis-output-dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}"
    done
  done
done

echo "Outstanding path_attn sweep complete for ${SOURCE_KEY}"
echo "Results root: $OUT_ROOT_ABS/$SOURCE_KEY"
