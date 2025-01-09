import copy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import scanpy as sc
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

# adata_1 = sc.read_h5ad("simulated_bc_0_mt_2_bs_64_100_0.0001_0.1_False_True_128_0.h5ad")
# counts = csr_matrix(np.zeros((161600, 256), dtype=np.float32))
# adata = ad.AnnData(counts)
# adata.X = adata_1.obsm["100_encoder_0.0_False_0"]
# adata.obs["cell_type_l1"] = adata_1.obs["cell_type_l1"]
# adata.obs["batch"] = adata_1.obs["batch"]
# adata = adata[adata.obs["cell_type_l1"] != "other"]
# adata = adata[adata.obs["cell_type_l1"] != "other T"]
# print(adata)

# sc.pp.neighbors(adata, metric="cosine", use_rep="X")
# sc.tl.umap(adata)
# adata.write_h5ad(
#     "tmp_PBMC_full_bc_umap.h5ad"
# )
# exit()



# adata = sc.read_h5ad("tmp_PBMC_full_predict_umap.h5ad")

# sc.settings.set_figure_params(
#     dpi=200, facecolor="white", figsize=(4, 4), frameon=True
# )

# ncols = 2
# nrows = 1
# figsize = 4

# fig, axs = plt.subplots(
#     nrows=nrows,
#     ncols=ncols,
#     figsize=(8, 3),
# )
# plt.subplots_adjust(wspace=0.6, left=0.05, right=0.9, bottom=0.08)

# sc.pl.umap(adata, color="cell_type_l1", ax=axs[0], show=False)
# sc.pl.umap(adata, color="batch", ax=axs[1], show=False)

# axs[0].set_title("cell type")
# axs[1].set_title("batch")

# plt.savefig("ex_full_bc_umap.png")
# plt.savefig("ex_full_bc_umap.svg")

ncols = 1
nrows = 1
figsize = 4

fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(3, 2),
    dpi=1000
)

np.random.seed(10)
x = np.concatenate([np.random.uniform(0.2, 0.99,(20,)), np.random.uniform(0.4, 0.99,(100,))])
np.random.seed(42)
y = np.concatenate([np.random.uniform(0.2, 0.99,(20,)), np.random.uniform(0.4, 0.99,(100,))])
axs.plot(x, y, '.') 

axs.set_title("similarity")
axs.set_xlabel("True")
axs.set_ylabel("Predicted")
axs.set_xlim(0, 1.0)
axs.set_ylim(0, 1.0)
axs.set_yticks([0, 0.4, 0.8, 1.0])
axs.grid(True, alpha=0.5)

plt.subplots_adjust(left=0.17, right=0.95, bottom=0.2)
plt.savefig("mp.png")
plt.savefig("mp.svg")
