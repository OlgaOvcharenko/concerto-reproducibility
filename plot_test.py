import copy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("tmp_cite_full.h5ad")

adata_test = copy.deepcopy(adata)
adata_test.obs["ds"] = pd.Series(["Query"] * adata_test.shape[0])
adata_test.obs["ds"][(adata_test.obs["batch"]!="P3") & (adata_test.obs["batch"]!="P8") & (adata_test.obs["batch"]!="P8")] = "Reference"
adata_test.obs["ds"][(adata_test.obs["batch"]=="P3") | (adata_test.obs["batch"]=="P8") | (adata_test.obs["batch"]=="P8")] = "Query"
adata_test.obs["ds"] = adata_test.obs["ds"].astype("category")

adata = adata[adata.obs["batch"]!="P3", ]
adata = adata[adata.obs["batch"]!="P8", ]
adata = adata[adata.obs["batch"]!="P5", ]
# adata = adata[adata.obs["batch"] not in ["P6", "P7", "P8"], ]
adata = adata[adata.obs["cell_type_l1"] != "other"]
adata = adata[adata.obs["cell_type_l1"] != "other T"]

adata = adata[((adata.obsm["X_umap"][:, 1] > 22)) != True]

sc.settings.set_figure_params(
    dpi=80, facecolor="white", figsize=(4, 4), frameon=True
)

ncols = 1
nrows = 1
figsize = 4

fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4, 3),
)
plt.subplots_adjust(wspace=0.6, left=0.06, right=0.7, bottom=0.08)

sc.pl.umap(adata, color="cell_type_l1", ax=axs, show=False)

plt.title("Reference (cell type)")

plt.savefig("ex_train.png")
plt.savefig("ex_train.svg")

sc.settings.set_figure_params(
    dpi=80, facecolor="white", figsize=(4, 4), frameon=True
)

ncols = 1
nrows = 1
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4, 3),
)
plt.subplots_adjust(wspace=0.6, left=0.06, right=0.7, bottom=0.08)

# sc.pl.umap(adata_test, color="ds", ax=axs, show=False)
sc.pl.umap(adata_test, ax=axs, show=False, color=["ds"], groups=["Query"])

legend_texts = axs.get_legend().get_texts()
for legend_text in legend_texts:
    if legend_text.get_text() == "NA":
        legend_text.set_text("Reference")

plt.title("")

plt.savefig("ex_train_query.png")
plt.savefig("ex_train_query.svg")

