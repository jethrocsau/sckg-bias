#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python}"
EMBEDDING_KEYS_CSV="${2:-ggd,dgi, grace}"
EMBED_SIZES_CSV="${3:-128}"
GAT_DEPTHS_CSV="${4:-1}"
KG_HOPS_CSV="${5:-2}"
KG_RELATION_LIMITS_CSV="${6:-2}"
NUM_FACTORS_CSV="${7:-1}"
LAMBDA_DIV="${8:-0.001}"
LAMBDA_SP="${9:-0.001}"
SPLIT_MODE="${10:-graph}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Preprocessing local star graph embeddings/splits once before sweep"
"$PYTHON_BIN" preprocessing/prepare_local_graph.py \
  --split-mode "$SPLIT_MODE" \
  --gnn-output data/star_graphs_gnn.pkl \
  --splits-output data/star_graph_edge_splits.json \
  --meta data/star_graphs_meta.json

"$PYTHON_BIN" test.py

IFS=',' read -r -a RAW_EMBEDDING_KEYS <<< "$EMBEDDING_KEYS_CSV"
IFS=',' read -r -a EMBED_SIZES <<< "$EMBED_SIZES_CSV"
IFS=',' read -r -a GAT_DEPTHS <<< "$GAT_DEPTHS_CSV"
IFS=',' read -r -a KG_HOPS <<< "$KG_HOPS_CSV"
IFS=',' read -r -a KG_RELATION_LIMITS <<< "$KG_RELATION_LIMITS_CSV"
IFS=',' read -r -a NUM_FACTORS_LIST <<< "$NUM_FACTORS_CSV"

EMBEDDING_KEYS=()
for key in "${RAW_EMBEDDING_KEYS[@]}"; do
  key="$(echo "$key" | xargs)"
  if [[ -z "$key" ]]; then continue; fi
  if [[ "$key" == "grace" || "$key" == "ggd" || "$key" == "dgi" ]]; then
    key="${key}_redaf"
  fi
  EMBEDDING_KEYS+=("$key")
done

echo "Using embedding keys: ${EMBEDDING_KEYS[*]}"

# Nested sweep loops
for source in "${EMBEDDING_KEYS[@]}"; do
  for embed_size in "${EMBED_SIZES[@]}"; do
    for gat_depth in "${GAT_DEPTHS[@]}"; do
      for num_factors in "${NUM_FACTORS_LIST[@]}"; do
        for kg_hops in "${KG_HOPS[@]}"; do
          for kg_relation_limit in "${KG_RELATION_LIMITS[@]}"; do
            
            kg_relation_limit="$(echo "$kg_relation_limit" | xargs)"
            [[ -z "$kg_relation_limit" ]] && kg_relation_limit="all"
            
            echo "==============================================================="
            echo "Running pipeline kg_graph=${source} embed=${embed_size} gat_depth=${gat_depth} num_factors=${num_factors} hops=${kg_hops} rel_limit=${kg_relation_limit} div=${LAMBDA_DIV} sp=${LAMBDA_SP}"
            echo "==============================================================="
            
            # Pass $num_factors as the 6th positional argument to the pipeline script
            bash scripts/run_variant_pipeline.sh "$source" "$PYTHON_BIN" "$embed_size" "$gat_depth" 1 "$num_factors" "$kg_hops" "$kg_relation_limit" "$LAMBDA_DIV" "$LAMBDA_SP" "$SPLIT_MODE"
            
          done
        done
      done
    done
  done
done

echo "Sweep complete. Aggregate summary: results/variant_runs/all_runs_summary.csv"