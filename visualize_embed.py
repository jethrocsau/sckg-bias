import scanpy as sc
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# 1. Load your generated embeddings
data_path = "data/scgpt_embeds/tahoe_embeddings_parquet.npz"
print(f"Loading embeddings from {data_path}...")
data = np.load(data_path, allow_pickle=True)

all_embeddings = []
all_metadata = []

# 2. Iterate and expand to match X rows with obs rows
for key in data.files:
    try:
        # Safely parse the key (cell_line, drug, plate)
        meta = ast.literal_eval(key) 
    except:
        continue

    emb = data[key]
    
    # Handle cases where emb might be a single vector or a stack of vectors
    if emb.ndim == 1:
        all_embeddings.append(emb)
        all_metadata.append(meta)
    else:
        # If it's a batch of embeddings for one condition, add meta for each
        for i in range(emb.shape[0]):
            all_embeddings.append(emb[i])
            all_metadata.append(meta)

# 3. Build AnnData
X = np.vstack(all_embeddings)
obs = pd.DataFrame(all_metadata, columns=["cell_line", "drug", "plate"])

print(f"Creating AnnData with {X.shape[0]} observations...")
embed_adata = sc.AnnData(X=X, obs=obs)

# 4. Standard Scanpy Pipeline
# Downsample for speed if needed (e.g., if X > 100k rows)
if embed_adata.n_obs > 50000:
    print("Downsampling for visualization speed...")
    sc.pp.subsample(embed_adata, n_obs=50000)

sc.pp.neighbors(embed_adata, use_rep="X", n_neighbors=15)
sc.tl.umap(embed_adata)

# 5. Plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sc.pl.umap(embed_adata, color="drug", ax=axes[0], show=False, title="UMAP by Drug")
sc.pl.umap(embed_adata, color="plate", ax=axes[1], show=False, title="UMAP by Plate")
plt.tight_layout()
plt.savefig("figs/embedding_umap_diagnostic.png")
print("Saved diagnostic UMAP to figs/embedding_umap_diagnostic.png")