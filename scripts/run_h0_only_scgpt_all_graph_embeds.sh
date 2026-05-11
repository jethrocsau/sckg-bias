#!/usr/bin/env bash
set -euo pipefail

# Run the shared h0-only baseline once, then mirror it into each KG-source folder.

PYTHON_BIN="${1:-python}"
SPLIT_MODE="${2:-graph}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-16}"
DEVICE="${5:-cuda}"
OUT_ROOT="${6:-results/baseline_comparison}"
EMBEDDING_SIZE="${7:-64}"
GAT_DEPTH="${8:-1}"
KG_SOURCE_KEYS_CSV="${9:-dgi, ggd, grace}"
LOCAL_EMBED_INPUT="${10:-data/scgpt_embeds/tahoe_embeddings_parquet*.npz}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$OUT_ROOT" = /* ]]; then
  OUT_ROOT_ABS="$OUT_ROOT"
else
  OUT_ROOT_ABS="$ROOT_DIR/$OUT_ROOT"
fi

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

mkdir -p "$OUT_ROOT_ABS"

echo "[1/3] Rebuilding local graph artifacts with split mode: ${SPLIT_MODE}"
"$PYTHON_BIN" preprocessing/prepare_local_graph.py \
  --input "$LOCAL_EMBED_INPUT" \
  --split-mode "$SPLIT_MODE" \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json

echo "[2/3] Auditing current splits"
"$PYTHON_BIN" test.py

IFS=',' read -r -a RAW_KG_SOURCE_KEYS <<< "$KG_SOURCE_KEYS_CSV"
KG_SOURCE_KEYS=()
for raw_key in "${RAW_KG_SOURCE_KEYS[@]}"; do
  key="$(normalize_embedding_key "$raw_key")"
  [[ -z "$key" ]] && continue
  PRIMEKG_DIR="$(resolve_primekg_dir "$key")"
  if [[ -d "$PRIMEKG_DIR" ]]; then
    KG_SOURCE_KEYS+=("$key")
  else
    echo "[warn] Skipping missing KG embedding source: $key ($PRIMEKG_DIR)"
  fi
done

if [[ ${#KG_SOURCE_KEYS[@]} -eq 0 ]]; then
  echo "No valid KG embedding sources found for: $KG_SOURCE_KEYS_CSV" >&2
  exit 1
fi

SHARED_RUN_DIR="$OUT_ROOT_ABS/_shared/h0_only_na_hopna_relall"
echo "[3/3] Running shared stage1 baseline variant=h0_only"
"$PYTHON_BIN" train_star_factor_gat.py \
  --run-training \
  --stage 1 \
  --model-variant h0_only \
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
  --analysis-output-dir "$SHARED_RUN_DIR" \
  --primekg-dir "$(resolve_primekg_dir "${KG_SOURCE_KEYS[0]}")"

for kg_source_key in "${KG_SOURCE_KEYS[@]}"; do
  linked_run_dir="$OUT_ROOT_ABS/${kg_source_key}/$(basename "$SHARED_RUN_DIR")"
  mkdir -p "$(dirname "$linked_run_dir")"
  rm -rf "$linked_run_dir"
  ln -sfnT "$SHARED_RUN_DIR" "$linked_run_dir"
  echo "Linked h0_only into ${kg_source_key}"
done

echo "Done: $SHARED_RUN_DIR"