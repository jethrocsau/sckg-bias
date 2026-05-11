#!/usr/bin/env bash
set -euo pipefail

# Baseline ablation sweep.
#
# Shared baseline architecture:
# - all baseline variants use the same 4-branch MLP input of width 4 * d_model
# - branches are [h0, target, context, kg_context]
# - h0 is always present; missing optional branches are replaced by learned embeddings of size d_model
#
# Variants:
# - h0_only                = [h0, learned_target, learned_context, learned_kg]
# - target_only            = [target, learned_context, learned_kg]
# - context_only           = [learned_target, context, learned_kg]
# - kg_context             = [learned_target, learned_context, kg]
# - target_plus_context    = [target, context, learned_kg]
# - context_plus_kg        = [learned_target, context, kg]
# - target_plus_kg         = [target, learned_context, kg]
# - target_plus_context_kg = [target, context, kg]
#
# Sweep controls:
# - KG_SOURCE_KEYS_CSV selects which PrimeKG embedding bank(s) to use
# - KG_METHODS_CSV selects the KG encoder/pooling method(s)
# - KG_HOPS and KG_RELATION_LIMIT control the KG neighborhood construction
#
# Note: KG_SOURCE_KEYS_CSV is about PrimeKG sources (grace/ggd/dgi/slgnn), not
# the Tahoe single-cell embeddings used to build the local star graph payload.

PYTHON_BIN="${1:-python}"
SPLIT_MODE="${2:-graph}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-16}"
DEVICE="${5:-cuda}"
OUT_ROOT="${6:-results/baseline_comparison}"
EMBEDDING_SIZE="${7:-64}"
GAT_DEPTH="${8:-1}"
KG_HOPS="${9:-3}"
KG_RELATION_LIMIT="${10:-1,2,4,6,8, all}"
KG_SOURCE_KEYS_CSV="${11:-dgi, ggd, grace}"
KG_METHODS_CSV="${12:-path_attn}"
LOCAL_EMBED_INPUT="${13:-data/scgpt_embeds/tahoe_embeddings_parquet*.npz}"

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

echo "[1/3] Rebuilding local graph artifacts with split mode: ${SPLIT_MODE}"
"$PYTHON_BIN" preprocessing/prepare_local_graph.py \
  --input "$LOCAL_EMBED_INPUT" \
  --split-mode "$SPLIT_MODE" \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json

echo "[2/3] Auditing current splits"
"$PYTHON_BIN" test.py

mkdir -p "$OUT_ROOT_ABS"

SUMMARY_ROWS=()
WEIGHT_SUMMARIES=()
PATH_WEIGHT_SUMMARIES=()
PREDICTION_SUMMARIES=()
PREDICTION_COMPARISONS=()
NON_KG_VARIANTS=(h0_only target_only context_only target_plus_context)
KG_VARIANTS=(kg_context context_plus_kg target_plus_kg target_plus_context_kg)
IFS=',' read -r -a RAW_KG_SOURCE_KEYS <<< "$KG_SOURCE_KEYS_CSV"
IFS=',' read -r -a KG_HOPS_LIST <<< "$KG_HOPS"
IFS=',' read -r -a KG_RELATION_LIMIT_LIST <<< "$KG_RELATION_LIMIT"
IFS=',' read -r -a RAW_KG_METHODS <<< "$KG_METHODS_CSV"

collect_run_outputs() {
  local run_dir="$1"
  SUMMARY_ROWS+=("$run_dir/experiment_summary.csv")
  if [[ -f "$run_dir/single_run.kg_relation_weight_summary.csv" ]]; then
    WEIGHT_SUMMARIES+=("$run_dir/single_run.kg_relation_weight_summary.csv")
  fi
  if [[ -f "$run_dir/single_run.kg_path_weight_summary.csv" ]]; then
    PATH_WEIGHT_SUMMARIES+=("$run_dir/single_run.kg_path_weight_summary.csv")
  fi
  if [[ -f "$run_dir/single_run.prediction_summary.csv" ]]; then
    PREDICTION_SUMMARIES+=("$run_dir/single_run.prediction_summary.csv")
  fi
  if [[ -f "$run_dir/single_run.prediction_comparison.csv" ]]; then
    PREDICTION_COMPARISONS+=("$run_dir/single_run.prediction_comparison.csv")
  fi
}

link_shared_run_for_source() {
  local shared_run_dir="$1"
  local kg_source_key="$2"
  local linked_run_dir="$OUT_ROOT_ABS/${kg_source_key}/$(basename "$shared_run_dir")"
  mkdir -p "$(dirname "$linked_run_dir")"
  rm -rf "$linked_run_dir"
  ln -sfnT "$shared_run_dir" "$linked_run_dir"
  collect_run_outputs "$linked_run_dir"
}

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

CONFIGURED_KG_METHODS=()
for raw_method in "${RAW_KG_METHODS[@]}"; do
  kg_method="$(echo "$raw_method" | xargs)"
  [[ -z "$kg_method" ]] && continue
  case "$kg_method" in
    mean_decay|gat|dgl_gat|path_attn)
      CONFIGURED_KG_METHODS+=("$kg_method")
      ;;
    *)
      echo "[warn] Skipping unsupported KG method: $kg_method"
      ;;
  esac
done

if [[ ${#CONFIGURED_KG_METHODS[@]} -eq 0 ]]; then
  echo "No valid KG methods found for: $KG_METHODS_CSV" >&2
  exit 1
fi

if [[ ${#KG_SOURCE_KEYS[@]} -eq 0 ]]; then
  echo "No valid KG embedding sources found for: $KG_SOURCE_KEYS_CSV" >&2
  exit 1
fi

for variant in "${NON_KG_VARIANTS[@]}"; do
  SHARED_RUN_DIR="$OUT_ROOT_ABS/_shared/${variant}_na_hopna_relall"
  echo "[3/3] Running shared stage1 baseline variant=${variant}"
  "$PYTHON_BIN" train_star_factor_gat.py \
    --run-training \
    --stage 1 \
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
    --analysis-output-dir "$SHARED_RUN_DIR" \
    --primekg-dir "$(resolve_primekg_dir "${KG_SOURCE_KEYS[0]}")"
  for kg_source_key in "${KG_SOURCE_KEYS[@]}"; do
    link_shared_run_for_source "$SHARED_RUN_DIR" "$kg_source_key"
  done
done

for kg_source_key in "${KG_SOURCE_KEYS[@]}"; do
  PRIMEKG_DIR="$(resolve_primekg_dir "$kg_source_key")"
  for variant in "${KG_VARIANTS[@]}"; do
    for kg_method in "${CONFIGURED_KG_METHODS[@]}"; do
      for kg_hops in "${KG_HOPS_LIST[@]}"; do
        kg_hops="$(echo "$kg_hops" | xargs)"
        [[ -z "$kg_hops" ]] && kg_hops="na"
        for kg_relation_limit in "${KG_RELATION_LIMIT_LIST[@]}"; do
          kg_relation_limit="$(echo "$kg_relation_limit" | xargs)"
          [[ -z "$kg_relation_limit" ]] && kg_relation_limit="all"

          RUN_DIR="$OUT_ROOT_ABS/${kg_source_key}/${variant}_${kg_method}_hop${kg_hops}_rel${kg_relation_limit}"
          echo "[3/3] Running kg_source=${kg_source_key} variant=${variant} kg=${kg_method} hops=${kg_hops} rel_limit=${kg_relation_limit}"
          EXTRA_ARGS=(--primekg-dir "$PRIMEKG_DIR" --precompute-kg-multihop --kg-hops "$kg_hops" --kg-embed-method "$kg_method")
          if [[ "$kg_relation_limit" != "all" ]]; then
            EXTRA_ARGS+=(--kg-relation-limit "$kg_relation_limit")
          fi
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
          collect_run_outputs "$RUN_DIR"
        done
      done
    done
  done
done

comparison_out="$OUT_ROOT_ABS/comparison_summary.csv"
comparison_header_written=0
for summary in "${SUMMARY_ROWS[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    kg_source="$(basename "$(dirname "$(dirname "$summary")")")"
    if [[ $comparison_header_written -eq 0 ]]; then
      header="$(head -n 1 "$summary")"
      echo "kg_source,variant,${header}" > "$comparison_out"
      comparison_header_written=1
    fi
    awk -F, -v kg_source="$kg_source" -v variant="$variant" 'NR==1{next} {print kg_source","variant","$0}' "$summary" >> "$comparison_out"
  fi
done

echo "kg_source,variant,relation_idx,relation_id,relation_name,weight_mean,weight_std,weight_min,weight_max" > "$OUT_ROOT_ABS/kg_weight_bias_summary.csv"
for summary in "${WEIGHT_SUMMARIES[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    kg_source="$(basename "$(dirname "$(dirname "$summary")")")"
    awk -F, -v kg_source="$kg_source" -v variant="$variant" 'NR==1{next} {print kg_source","variant","$0}' "$summary" >> "$OUT_ROOT_ABS/kg_weight_bias_summary.csv"
  fi
done

echo "kg_source,variant,path_idx,path_hops,relation_seq,weight_mean,weight_std,weight_min,weight_max" > "$OUT_ROOT_ABS/kg_path_bias_summary.csv"
for summary in "${PATH_WEIGHT_SUMMARIES[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    kg_source="$(basename "$(dirname "$(dirname "$summary")")")"
    awk -F, -v kg_source="$kg_source" -v variant="$variant" 'NR==1{next} {print kg_source","variant","$0}' "$summary" >> "$OUT_ROOT_ABS/kg_path_bias_summary.csv"
  fi
done

echo "kg_source,variant,split,num_samples,acc,confidence_mean,confidence_correct_mean,confidence_incorrect_mean,num_unique_true,num_unique_pred" > "$OUT_ROOT_ABS/prediction_summary.csv"
for summary in "${PREDICTION_SUMMARIES[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    kg_source="$(basename "$(dirname "$(dirname "$summary")")")"
    awk -F, -v kg_source="$kg_source" -v variant="$variant" 'NR==1{next} {print kg_source","variant","$0}' "$summary" >> "$OUT_ROOT_ABS/prediction_summary.csv"
  fi
done

echo "kg_source,variant,split,true_edge_class_idx,true_edge_name,pred_edge_class_idx,pred_edge_name,count,confidence_mean,confidence_std,correct_rate" > "$OUT_ROOT_ABS/prediction_comparison.csv"
for summary in "${PREDICTION_COMPARISONS[@]}"; do
  if [[ -f "$summary" ]]; then
    variant="$(basename "$(dirname "$summary")")"
    kg_source="$(basename "$(dirname "$(dirname "$summary")")")"
    awk -F, -v kg_source="$kg_source" -v variant="$variant" 'NR==1{next} {print kg_source","variant","$0}' "$summary" >> "$OUT_ROOT_ABS/prediction_comparison.csv"
  fi
done

echo "Baseline comparison complete: ${OUT_ROOT_ABS}"
echo "Summary CSV: ${OUT_ROOT_ABS}/comparison_summary.csv"
echo "KG bias summary CSV: ${OUT_ROOT_ABS}/kg_weight_bias_summary.csv"
echo "KG path summary CSV: ${OUT_ROOT_ABS}/kg_path_bias_summary.csv"
echo "Prediction summary CSV: ${OUT_ROOT_ABS}/prediction_summary.csv"
echo "Prediction comparison CSV: ${OUT_ROOT_ABS}/prediction_comparison.csv"