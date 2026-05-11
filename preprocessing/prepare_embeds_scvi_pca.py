import ast
import argparse
import gc
import glob
import json
import os
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
import torch

try:
    import scvi

    HAS_SCVI = True
except Exception:
    scvi = None
    HAS_SCVI = False

SAVE_INTERVAL = 100000
BATCH_SIZE = 32
MAX_RETRIES = 3
DEFAULT_RESUME_FILE = "train-00000-of-03388.parquet"
DEFAULT_PARQUET_SHARD_SIZE = 400
DEFAULT_MANIFEST_SUFFIX = ".shards_manifest.json"
DEFAULT_EMBED_METHOD = "auto"
DEFAULT_LATENT_DIM = 64
DEFAULT_SCVI_MAX_EPOCHS = 20
DEFAULT_SCVI_BATCH_SIZE = 256
DEFAULT_SCVI_QUERY_MAX_EPOCHS = 10
DEFAULT_SCVI_HUB_REPO = "vevotx/Tahoe-100M-SCVI-v1"


def get_paths():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    root_dir = repo_root.parent
    parquet_dir = (root_dir / "GeneJEPA/hf_data_cache/data/data").resolve()
    output_path = repo_root / "data" / "x1_embeds" / "scvi_embeddings_parquet.npz"
    map_path = repo_root / "tahoe_to_primekg_map.csv"
    return script_dir, parquet_dir, output_path, map_path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SCVI/PCA embeddings from parquet batches.")
    parser.add_argument(
        "--embed-method",
        type=str,
        default=DEFAULT_EMBED_METHOD,
        choices=["auto", "scvi", "pca"],
        help="Embedding backend: auto (prefer SCVI, fallback to PCA), scvi, or pca.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
        help="Embedding dimension to produce for each cell.",
    )
    parser.add_argument(
        "--scvi-max-epochs",
        type=int,
        default=DEFAULT_SCVI_MAX_EPOCHS,
        help="Max epochs for SCVI training per batch.",
    )
    parser.add_argument(
        "--scvi-batch-size",
        type=int,
        default=DEFAULT_SCVI_BATCH_SIZE,
        help="Mini-batch size used during SCVI training.",
    )
    parser.add_argument(
        "--use-scvi-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When SCVI is selected, use pretrained SCVI Hub model for query embeddings.",
    )
    parser.add_argument(
        "--scvi-hub-repo",
        type=str,
        default=DEFAULT_SCVI_HUB_REPO,
        help="Hugging Face Hub repo for pretrained SCVI model.",
    )
    parser.add_argument(
        "--scvi-hub-cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for SCVI Hub model download.",
    )
    parser.add_argument(
        "--scvi-query-max-epochs",
        type=int,
        default=DEFAULT_SCVI_QUERY_MAX_EPOCHS,
        help="Max epochs to adapt SCVI query model when using Hub model.",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default=DEFAULT_RESUME_FILE,
        help="Parquet filename to resume from (inclusive).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="0-based parquet file index to resume from (inclusive). Overrides --start-file when provided.",
    )
    parser.add_argument(
        "--parquet-shard-size",
        type=int,
        default=DEFAULT_PARQUET_SHARD_SIZE,
        help="How many parquet files to process in this run.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index after resume point.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output NPZ path. If omitted, defaults to data/x1_embeds/scvi_embeddings_parquet.npz.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow writing directly to --output-path even if it already exists.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional shard-manifest JSON path. Defaults next to output NPZ.",
    )
    parser.add_argument(
        "--disable-manifest",
        action="store_true",
        help="Disable shard-manifest skip/record behavior.",
    )
    parser.add_argument(
        "--center-drugs",
        type=str,
        default="DMSO_TF",
        help=(
            "Comma-separated center/control drugs to always include in embedding extraction "
            "in addition to mapped target drugs."
        ),
    )
    parser.add_argument(
        "--require-complete-stars",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, first scan parquet files and keep only (cell_line, plate) "
            "groups that contain all required drugs (target drugs + center drugs)."
        ),
    )
    parser.add_argument(
        "--require-center-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, keep only (cell_line, plate) groups that contain at least one "
            "center/control drug before embedding."
        ),
    )
    return parser.parse_args()


def resolve_processing_window(parquet_files, start_file, start_index, shard_size, shard_index):
    if start_index is not None:
        if start_index < 0 or start_index >= len(parquet_files):
            raise ValueError(f"--start-index out of range: {start_index} (files={len(parquet_files)})")
        resume_start = start_index
    else:
        target_name = Path(start_file).name
        name_to_idx = {Path(path).name: idx for idx, path in enumerate(parquet_files)}
        if target_name not in name_to_idx:
            raise ValueError(f"--start-file not found in parquet directory: {target_name}")
        resume_start = name_to_idx[target_name]

    if shard_size <= 0:
        raise ValueError("--parquet-shard-size must be > 0")
    if shard_index < 0:
        raise ValueError("--shard-index must be >= 0")

    shard_start = resume_start + shard_index * shard_size
    shard_end = min(shard_start + shard_size, len(parquet_files))
    if shard_start >= len(parquet_files):
        return resume_start, shard_start, shard_end, []

    return resume_start, shard_start, shard_end, parquet_files[shard_start:shard_end]


def build_safe_output_path(requested_output_path: Path, shard_start: int, shard_end: int, overwrite_output: bool):
    if overwrite_output or not requested_output_path.exists():
        return requested_output_path

    stem = requested_output_path.stem
    suffix = requested_output_path.suffix
    safe_name = f"{stem}.resume_{shard_start:05d}_{max(shard_end - 1, shard_start):05d}{suffix}"
    return requested_output_path.with_name(safe_name)


def resolve_manifest_path(output_path: Path, manifest_path_arg: str | None):
    if manifest_path_arg:
        return Path(manifest_path_arg).expanduser().resolve()
    stem = output_path.stem
    return output_path.with_name(f"{stem}{DEFAULT_MANIFEST_SUFFIX}")


def load_shard_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {"completed_shards": []}

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        return {"completed_shards": []}

    completed = manifest.get("completed_shards", [])
    if not isinstance(completed, list):
        completed = []
    manifest["completed_shards"] = completed
    return manifest


def save_shard_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def make_shard_id(shard_start: int, shard_end: int):
    return f"{shard_start:05d}:{max(shard_end - 1, shard_start):05d}"


def mark_shard_completed(manifest: dict, shard_id: str, shard_start: int, shard_end: int, file_count: int):
    completed = manifest.setdefault("completed_shards", [])
    for entry in completed:
        if isinstance(entry, dict) and entry.get("id") == shard_id:
            return

    completed.append(
        {
            "id": shard_id,
            "start_index": shard_start,
            "end_index_inclusive": max(shard_end - 1, shard_start),
            "file_count": file_count,
            "completed_at": int(time.time()),
        }
    )


def parse_target_drugs(map_path: Path, center_drugs: set[str] | None = None):
    if not map_path.exists():
        raise FileNotFoundError(f"Target drug mapping file not found: {map_path}")

    mapping_df = pd.read_csv(map_path)
    if "drug" not in mapping_df.columns:
        raise ValueError(f"Expected 'drug' column in {map_path}")

    target_drugs = set(mapping_df["drug"].dropna().astype(str).str.strip())
    target_drugs = {drug for drug in target_drugs if drug}

    if center_drugs:
        target_drugs.update({str(drug).strip() for drug in center_drugs if str(drug).strip()})

    if not target_drugs:
        raise ValueError(f"No target drugs found in {map_path}")

    return target_drugs


def load_model_bundle(
    embed_method: str,
    latent_dim: int,
    scvi_max_epochs: int,
    scvi_batch_size: int,
    device: str,
    use_scvi_hub: bool,
    scvi_hub_repo: str,
    scvi_hub_cache_dir: str | None,
    scvi_query_max_epochs: int,
):
    effective_method = embed_method
    if embed_method == "auto":
        effective_method = "scvi" if HAS_SCVI else "pca"
    if effective_method == "scvi" and not HAS_SCVI:
        raise ImportError(
            "scvi-tools is not installed. Use --embed-method pca or install scvi-tools."
        )

    hub_scvi_model = None
    if effective_method == "scvi" and use_scvi_hub:
        try:
            hub_cache = Path(scvi_hub_cache_dir).expanduser().resolve() if scvi_hub_cache_dir else None
            print(f"Loading SCVI Hub model from {scvi_hub_repo}...")
            hub_model = scvi.hub.HubModel.pull_from_huggingface_hub(
                repo_name=scvi_hub_repo,
                cache_dir=str(hub_cache) if hub_cache else None,
            )
            hub_scvi_model = hub_model.model
            hub_latent_dim = int(getattr(hub_scvi_model.module, "n_latent", latent_dim))
            print(f"Loaded SCVI Hub model (n_latent={hub_latent_dim}).")
            if hub_latent_dim != int(latent_dim):
                print(
                    "Note: requested --latent-dim "
                    f"{latent_dim} differs from hub model latent {hub_latent_dim}; "
                    "embeddings will be padded/truncated to requested size."
                )
        except Exception as exc:
            print(f"Warning: failed to load SCVI Hub model ({exc}); falling back to local SCVI training.")

    print(f"Embedding backend: {effective_method} (requested={embed_method})")
    return {
        "embed_method": effective_method,
        "device": device,
        "embsize": int(latent_dim),
        "scvi_max_epochs": int(scvi_max_epochs),
        "scvi_batch_size": int(scvi_batch_size),
        "scvi_query_max_epochs": int(scvi_query_max_epochs),
        "scvi_hub_model": hub_scvi_model,
    }


def embed_adata_with_x1(adata_batch: ad.AnnData, model_bundle):
    """Generate embeddings using SCVI or PCA."""
    if adata_batch.n_obs == 0:
        return np.zeros((0, model_bundle["embsize"]), dtype=np.float32)

    method = model_bundle["embed_method"]
    embsize = model_bundle["embsize"]

    try:
        if method == "scvi":
            adata_work = adata_batch.copy()
            hub_scvi_model = model_bundle.get("scvi_hub_model")

            if hub_scvi_model is not None:
                if "counts" not in adata_work.layers:
                    adata_work.layers["counts"] = adata_work.X.copy()

                scvi.model.SCVI.prepare_query_anndata(adata_work, hub_scvi_model)
                query_model = scvi.model.SCVI.load_query_data(adata_work, hub_scvi_model)
                if model_bundle["scvi_query_max_epochs"] > 0 and adata_work.n_obs > 1:
                    query_model.train(
                        max_epochs=model_bundle["scvi_query_max_epochs"],
                        batch_size=min(model_bundle["scvi_batch_size"], max(1, adata_work.n_obs)),
                        train_size=1.0,
                    )
                embeddings = query_model.get_latent_representation(adata_work).astype(np.float32)
            else:
                scvi.model.SCVI.setup_anndata(adata_work)
                scvi_model = scvi.model.SCVI(adata_work, n_latent=embsize)
                scvi_model.train(
                    max_epochs=model_bundle["scvi_max_epochs"],
                    batch_size=min(model_bundle["scvi_batch_size"], max(1, adata_work.n_obs)),
                    train_size=1.0,
                )
                embeddings = scvi_model.get_latent_representation(adata_work).astype(np.float32)
        else:
            adata_work = adata_batch.copy()
            min_dim = min(adata_work.n_obs, adata_work.n_vars)
            if min_dim <= 1:
                fallback = np.asarray(adata_work.X.sum(axis=1), dtype=np.float32).reshape(-1, 1)
                if fallback.shape[1] < embsize:
                    pad = np.zeros((fallback.shape[0], embsize - fallback.shape[1]), dtype=np.float32)
                    embeddings = np.concatenate([fallback, pad], axis=1)
                else:
                    embeddings = fallback[:, :embsize]
            else:
                n_pca = max(1, min(embsize, adata_work.n_obs - 1, adata_work.n_vars - 1))
                sc.tl.pca(adata_work, n_comps=n_pca, svd_solver="arpack")
                pca_embeddings = adata_work.obsm["X_pca"].astype(np.float32)
                if pca_embeddings.shape[1] < embsize:
                    pad = np.zeros((pca_embeddings.shape[0], embsize - pca_embeddings.shape[1]), dtype=np.float32)
                    embeddings = np.concatenate([pca_embeddings, pad], axis=1)
                else:
                    embeddings = pca_embeddings
    except Exception as exc:
        print(f"Error during embedding ({method}): {exc}")
        return np.zeros((adata_batch.n_obs, embsize), dtype=np.float32)

    if embeddings.shape[1] != embsize:
        if embeddings.shape[1] < embsize:
            pad = np.zeros((embeddings.shape[0], embsize - embeddings.shape[1]), dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=1)
        else:
            embeddings = embeddings[:, :embsize]

    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings


def load_existing_embeddings(output_path: Path, target_drugs, allowed_star_groups: set[tuple[str, str]] | None = None):
    condition_embeddings_acc = defaultdict(list)
    if not output_path.exists():
        return condition_embeddings_acc

    print(f"Loading existing embeddings from {output_path}...")
    skipped_non_target = 0
    with np.load(output_path, allow_pickle=True) as data:
        for key in data.files:
            try:
                tuple_key = ast.literal_eval(key)
            except (SyntaxError, ValueError):
                skipped_non_target += 1
                continue

            if not isinstance(tuple_key, tuple) or len(tuple_key) < 2:
                skipped_non_target += 1
                continue

            if str(tuple_key[1]) not in target_drugs:
                skipped_non_target += 1
                continue

            if allowed_star_groups is not None:
                if len(tuple_key) < 3:
                    skipped_non_target += 1
                    continue
                group_key = (str(tuple_key[0]), str(tuple_key[2]))
                if group_key not in allowed_star_groups:
                    skipped_non_target += 1
                    continue

            condition_embeddings_acc[tuple_key] = [data[key]]

    print(f"Loaded {len(condition_embeddings_acc)} condition groups.")
    if skipped_non_target > 0:
        print(f"Skipped {skipped_non_target} non-target groups from existing checkpoint.")
    return condition_embeddings_acc


def save_checkpoints(acc_dict, path: Path, total_records_read: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_dict = {}
    for key, arrays in acc_dict.items():
        string_key = str(key)
        if len(arrays) == 1 and isinstance(arrays[0], np.ndarray):
            flat_dict[string_key] = arrays[0]
        else:
            flat_dict[string_key] = np.vstack(arrays)
    np.savez_compressed(path, **flat_dict)
    print(f"Checkpoint saved to {path} at {total_records_read} records.")


def create_anndata_from_batch(df: pd.DataFrame):
    """
    Create AnnData object from parquet batch.
    Genes and expressions are stored as lists in the dataframe.
    """
    # Collect all gene names
    all_genes = set()
    for genes_list in df["genes"]:
        if genes_list is not None and len(genes_list) > 0:
            # Skip first element (CLS token from scGPT preprocessing)
            all_genes.update(genes_list[1:])
    
    gene_names = sorted(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Build sparse CSR matrix
    data, indices, indptr = [], [], [0]
    
    for _, row in df.iterrows():
        genes = row["genes"]
        expressions = row["expressions"]
        
        if genes is not None and expressions is not None and len(genes) > 0 and len(expressions) > 0:
            # Skip first element (CLS token)
            genes = genes[1:]
            expressions = expressions[1:]
        
        for gene, expression in zip(genes, expressions):
            if gene in gene_to_idx:
                indices.append(gene_to_idx[gene])
                data.append(float(expression))
        
        indptr.append(len(data))
    
    from scipy.sparse import csr_matrix
    
    matrix = csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(df), len(gene_names)),
    )
    
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))
    var["index"] = var.index
    var.index = var.index.astype(str)
    
    obs_columns = [col for col in ["cell_line_id", "drug", "plate"] if col in df.columns]
    if obs_columns:
        obs = df[obs_columns].copy()
    else:
        obs = pd.DataFrame(index=np.arange(len(df)))
    
    obs.index = obs.index.astype(str)
    
    return ad.AnnData(X=matrix, obs=obs, var=var)


def build_condition_key(obs_row):
    cell_line = obs_row.get("cell_line_id", "NA")
    drug = obs_row.get("drug", "NA")
    plate = obs_row.get("plate", "NA")
    return str(cell_line), str(drug), str(plate)


def process_parquet_file(
    parquet_file: Path,
    target_drugs,
    allowed_star_groups: set[tuple[str, str]] | None,
    condition_embeddings_acc,
    model_bundle,
    total_records_read: int,
    output_path: Path,
):
    pq_file = pq.ParquetFile(parquet_file)
    columns = ["genes", "expressions", "drug", "plate", "cell_line_id"]
    schema_columns = set(pq_file.schema_arrow.names)
    available_columns = [c for c in columns if c in schema_columns]

    if "genes" not in available_columns or "expressions" not in available_columns:
        print(f"Skipping {parquet_file.name}: missing genes/expressions columns")
        return total_records_read

    for record_batch in pq_file.iter_batches(batch_size=BATCH_SIZE, columns=available_columns):
        batch_df = record_batch.to_pandas()

        if "drug" not in batch_df.columns:
            total_records_read += len(batch_df)
            continue

        mask = batch_df["drug"].astype(str).isin(target_drugs)
        if allowed_star_groups is not None:
            if "cell_line_id" not in batch_df.columns or "plate" not in batch_df.columns:
                total_records_read += len(batch_df)
                continue
            group_mask = [
                (str(cell_line), str(plate)) in allowed_star_groups
                for cell_line, plate in zip(batch_df["cell_line_id"], batch_df["plate"])
            ]
            mask = mask & np.asarray(group_mask, dtype=bool)
        if not mask.any():
            total_records_read += len(batch_df)
            continue
        batch_df = batch_df[mask].reset_index(drop=True)

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                adata_batch = create_anndata_from_batch(batch_df)
                if adata_batch.n_obs == 0:
                    success = True
                    break
                
                # Tahoe 100m is already normalized
                cell_embeddings = embed_adata_with_x1(adata_batch, model_bundle)

                for i in range(adata_batch.n_obs):
                    obs_row = adata_batch.obs.iloc[i]
                    key = build_condition_key(obs_row)
                    condition_embeddings_acc[key].append(cell_embeddings[i : i + 1])

                success = True
                break
            except Exception as exc:
                print(f"Error on attempt {attempt + 1} in file {parquet_file.name}: {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    print(f"Failed batch in {parquet_file.name} after {MAX_RETRIES} attempts.")

        total_records_read += len(batch_df)

        if total_records_read % SAVE_INTERVAL < len(batch_df) and total_records_read >= SAVE_INTERVAL:
            save_checkpoints(condition_embeddings_acc, output_path, total_records_read)
            gc.collect()

        if not success:
            gc.collect()

    return total_records_read


def collect_complete_star_groups(parquet_files, required_drugs: set[str]):
    group_to_drugs = defaultdict(set)

    for parquet_file in parquet_files:
        pq_file = pq.ParquetFile(parquet_file)
        columns = ["drug", "plate", "cell_line_id"]
        schema_columns = set(pq_file.schema_arrow.names)
        if not set(columns).issubset(schema_columns):
            continue

        for record_batch in pq_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            batch_df = record_batch.to_pandas()
            if batch_df.empty:
                continue

            batch_df = batch_df.dropna(subset=["drug", "plate", "cell_line_id"])
            if batch_df.empty:
                continue

            batch_df = batch_df[batch_df["drug"].astype(str).isin(required_drugs)]
            if batch_df.empty:
                continue

            dedup = (
                batch_df[["cell_line_id", "plate", "drug"]]
                .astype(str)
                .drop_duplicates()
            )
            for row in dedup.itertuples(index=False):
                group_to_drugs[(str(row.cell_line_id), str(row.plate))].add(str(row.drug))

    complete_groups = {
        group_key
        for group_key, seen_drugs in group_to_drugs.items()
        if required_drugs.issubset(seen_drugs)
    }
    return complete_groups, len(group_to_drugs)


def collect_center_star_groups(parquet_files, center_drugs: set[str]):
    groups_with_center = set()
    groups_seen = set()

    for parquet_file in parquet_files:
        pq_file = pq.ParquetFile(parquet_file)
        columns = ["drug", "plate", "cell_line_id"]
        schema_columns = set(pq_file.schema_arrow.names)
        if not set(columns).issubset(schema_columns):
            continue

        for record_batch in pq_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            batch_df = record_batch.to_pandas()
            if batch_df.empty:
                continue

            batch_df = batch_df.dropna(subset=["drug", "plate", "cell_line_id"])
            if batch_df.empty:
                continue

            dedup = batch_df[["cell_line_id", "plate", "drug"]].astype(str).drop_duplicates()
            for row in dedup.itertuples(index=False):
                group_key = (str(row.cell_line_id), str(row.plate))
                groups_seen.add(group_key)
                if str(row.drug) in center_drugs:
                    groups_with_center.add(group_key)

    return groups_with_center, len(groups_seen)


def main():
    args = parse_args()
    script_dir, parquet_dir, output_path, map_path = get_paths()
    if args.output_path:
        output_candidate = Path(args.output_path).expanduser()
        if output_candidate.suffix.lower() != ".npz":
            output_candidate = output_candidate.with_suffix(".npz")

        if output_candidate.is_absolute():
            output_path = output_candidate.resolve()
        elif output_candidate.parent == Path("."):
            output_path = (script_dir / "data" / output_candidate.name).resolve()
        else:
            output_path = (script_dir / output_candidate).resolve()

    all_parquet_files = sorted(glob.glob(str(parquet_dir / "*.parquet")))

    if not all_parquet_files:
        raise FileNotFoundError(f"No parquet files found at {parquet_dir}")

    resume_start, shard_start, shard_end, parquet_files = resolve_processing_window(
        parquet_files=all_parquet_files,
        start_file=args.start_file,
        start_index=args.start_index,
        shard_size=args.parquet_shard_size,
        shard_index=args.shard_index,
    )

    if not parquet_files:
        print(
            f"No parquet files selected for this run. resume_start={resume_start}, "
            f"shard_index={args.shard_index}, shard_size={args.parquet_shard_size}."
        )
        return

    effective_output_path = build_safe_output_path(
        requested_output_path=output_path,
        shard_start=shard_start,
        shard_end=shard_end,
        overwrite_output=args.overwrite_output,
    )

    if effective_output_path != output_path:
        print(f"Existing output preserved at {output_path}")
        print(f"Writing resumed shard output to {effective_output_path}")

    shard_id = make_shard_id(shard_start, shard_end)
    manifest_path = resolve_manifest_path(effective_output_path, args.manifest_path)
    shard_manifest = None
    if not args.disable_manifest:
        shard_manifest = load_shard_manifest(manifest_path)
        completed_shards = shard_manifest.get("completed_shards", [])
        already_done = any(
            isinstance(entry, dict) and entry.get("id") == shard_id for entry in completed_shards
        )
        if already_done:
            print(f"Shard {shard_id} already completed per manifest: {manifest_path}")
            print("Skipping processing for this shard.")
            return
        print(f"Using shard manifest: {manifest_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")
    model_bundle = load_model_bundle(
        embed_method=args.embed_method,
        latent_dim=args.latent_dim,
        scvi_max_epochs=args.scvi_max_epochs,
        scvi_batch_size=args.scvi_batch_size,
        device=device,
        use_scvi_hub=args.use_scvi_hub,
        scvi_hub_repo=args.scvi_hub_repo,
        scvi_hub_cache_dir=args.scvi_hub_cache_dir,
        scvi_query_max_epochs=args.scvi_query_max_epochs,
    )

    center_drugs = {d.strip() for d in args.center_drugs.split(",") if d.strip()}
    target_drugs = parse_target_drugs(map_path, center_drugs=center_drugs)
    print(
        f"Filtering to {len(target_drugs)} drugs from {map_path.name} "
        f"(includes {len(center_drugs)} center/control drugs)."
    )

    allowed_star_groups = None
    if args.require_complete_stars:
        print("Scanning full parquet list for complete (cell_line, plate) stars...")
        allowed_star_groups, total_groups_seen = collect_complete_star_groups(all_parquet_files, target_drugs)
        print(
            f"Found {len(allowed_star_groups)} complete stars out of {total_groups_seen} observed "
            "(requires full target+center drug coverage)."
        )
        if not allowed_star_groups:
            raise ValueError(
                "No complete stars found in selected parquet files. "
                "Use a larger sample, different shard window, or disable with --no-require-complete-stars."
            )
    elif args.require_center_pairs:
        print("Scanning full parquet list for (cell_line, plate) groups with center/control drugs...")
        allowed_star_groups, total_groups_seen = collect_center_star_groups(all_parquet_files, center_drugs)
        print(
            f"Found {len(allowed_star_groups)} center-qualified groups out of {total_groups_seen} observed."
        )
        if not allowed_star_groups:
            raise ValueError(
                "No (cell_line, plate) groups with center/control drugs were found. "
                "Check --center-drugs or disable with --no-require-center-pairs."
            )

    seed_path = output_path if output_path.exists() else effective_output_path
    condition_embeddings_acc = load_existing_embeddings(
        seed_path,
        target_drugs,
        allowed_star_groups=allowed_star_groups,
    )
    total_records_read = 0

    print(
        f"Processing shard files [{shard_start}:{shard_end}) from {parquet_dir} "
        f"({len(parquet_files)} files this run)..."
    )
    for parquet_file in parquet_files:
        print(f"Processing {Path(parquet_file).name}...")
        total_records_read = process_parquet_file(
            Path(parquet_file),
            target_drugs,
            allowed_star_groups,
            condition_embeddings_acc,
            model_bundle,
            total_records_read,
            effective_output_path,
        )

    save_checkpoints(condition_embeddings_acc, effective_output_path, total_records_read)

    if shard_manifest is not None:
        mark_shard_completed(
            manifest=shard_manifest,
            shard_id=shard_id,
            shard_start=shard_start,
            shard_end=shard_end,
            file_count=len(parquet_files),
        )
        save_shard_manifest(manifest_path, shard_manifest)
        print(f"Marked shard {shard_id} completed in manifest.")

    print(f"Finished processing {total_records_read} records.")


if __name__ == "__main__":
    main()
