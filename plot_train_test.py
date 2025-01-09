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

# adata_train = sc.read_h5ad("simulated_qr_train_0_mt_2_bs_64_100_0.0001_0.1_False_True_128_0.h5ad")
# adata_test = sc.read_h5ad("simulated_qr_test_0_mt_2_bs_64_100_0.0001_0.1_False_True_128_0.h5ad")

# counts = csr_matrix(np.zeros((98944 + 62656, 256), dtype=np.float32))
# adata = ad.AnnData(counts)
# adata.X[0:98944, :] = adata_train.obsm["train_100_encoder_0.0_False_0"]
# adata.X[98944:, :] = adata_test.obsm["test_100_encoder_0.0_False_0"]

# # adata_test.obs["ds"] = adata_test.obs["ds"].astype("category")
# adata.obs["ds"] = pd.Series((["Reference"] * 98944) + (["Query"] * 62656)).astype("category")
# adata.obs["cell_type_test"] = ["Reference"] * 98944 + adata_test.obs["cell_type_l1"].tolist()
# adata.obs["cell_type_test"] = adata.obs["cell_type_test"].astype("category")

# adata.obs["cell_type_train"] = adata_train.obs["cell_type_l1"].tolist() + ["Query"] * 62656
# adata.obs["cell_type_train"] = adata.obs["cell_type_train"].astype("category")

# adata.obs["cell_type_l1"] = adata_train.obs["cell_type_l1"].tolist() + adata_test.obs["cell_type_l1"].tolist()
# adata.obs["cell_type_l1"] = adata.obs["cell_type_l1"].astype("category")

# adata.obs["batch"] = adata_train.obs["batch"].tolist() + adata_test.obs["batch"].tolist()
# adata.obs["batch"] = adata.obs["batch"].astype("category")

# adata = adata[adata.obs["cell_type_train"] != "other"]
# adata = adata[adata.obs["cell_type_train"] != "other T"]
# adata = adata[adata.obs["cell_type_test"] != "other"]
# adata = adata[adata.obs["cell_type_test"] != "other T"]

# print(adata)

# sc.pp.neighbors(adata, metric="cosine", use_rep="X")
# sc.tl.umap(adata)
# adata.write_h5ad(
#     "tmp_PBMC_full_predict_umap.h5ad"
# )
# exit()
adata = sc.read_h5ad("tmp_PBMC_full_predict_umap.h5ad")
print(adata.obs["cell_type_train"])

sc.settings.set_figure_params(
    dpi=80, facecolor="white", figsize=(4, 4), frameon=True
)

ncols = 2
nrows = 1
figsize = 4

fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8, 3),
)
plt.subplots_adjust(wspace=0.6, left=0.03, right=0.85, bottom=0.08)

sc.pl.umap(adata, color="cell_type_train", ax=axs[0], show=False, groups=['B', 'CD4 T', 'CD8 T', 'DC', 'Monocytes', 'NK'])
axs[0].set_title("Reference (cell type)")

legend_texts = axs[0].get_legend().get_texts()
for legend_text in legend_texts:
    if legend_text.get_text() == "NA":
        legend_text.set_text("Query")

sc.pl.umap(adata, color="cell_type_test", ax=axs[1], show=False, groups=['B', 'CD4 T', 'CD8 T', 'DC', 'Monocytes', 'NK'])
axs[1].set_title("Query (cell type)")

legend_texts = axs[1].get_legend().get_texts()
for legend_text in legend_texts:
    if legend_text.get_text() == "NA":
        legend_text.set_text("Reference")

plt.savefig("ex_full_predict_umap.png")
plt.savefig("ex_full_predict_umap.svg")
