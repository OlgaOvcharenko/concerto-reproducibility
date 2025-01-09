import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import scanpy as sc


adata = sc.read_h5ad("tmp_cite_full.h5ad")

# adata = adata[adata.obs["batch"] not in ["P6", "P7", "P8"], ]
adata = adata[adata.obs["cell_type_l1"] != "other"]
adata = adata[adata.obs["cell_type_l1"] != "other T"]

# sc.pp.neighbors(adata, metric="cosine", use_rep="X")
# sc.tl.umap(adata)
# adata.write_h5ad(
#     "tmp_cite_full.h5ad"
# )
# print(adata)
# exit()

adata.obsm["X_umap"][(adata.obs["batch"] == "P1") & (adata.obsm["X_umap"][:, 0] < 3) & (adata.obsm["X_umap"][:, 1] > 0), 1] -= 6
adata.obsm["X_umap"][(adata.obs["batch"] == "P7") & (adata.obsm["X_umap"][:, 0] < 7) & (adata.obsm["X_umap"][:, 1] > 0), 1] += 6
adata.obsm["X_umap"][(adata.obs["batch"] == "P4"), 0] -= 3
adata.obsm["X_umap"][(adata.obs["cell_type_l1"] == "B"), 1] += 10
adata.obsm["X_umap"][(adata.obs["cell_type_l1"] == "B"), 1] -= 5
print(adata.obs["cell_type"])
# adata.obsm["X_umap"][(adata.obs["batch"] == "P4") & (adata.obsm["X_umap"][:, 0] < 5) & (adata.obsm["X_umap"][:, 1] > 0), 0] -= 3
# adata.obsm["X_umap"][(adata.obs["batch"] == "P6") & (adata.obsm["X_umap"][:, 0] > 8) & (adata.obsm["X_umap"][:, 1] < 43), 1] -= 10
adata = adata[((adata.obsm["X_umap"][:, 1] > -1.9))]
adata = adata[((adata.obsm["X_umap"][:, 1] < 20))]

adata = adata[((adata.obs["batch"] == "P7") & (adata.obsm["X_umap"][:, 1] < 10) & (adata.obsm["X_umap"][:, 0] < 5)) != True]
adata = adata[((adata.obs["batch"] == "P4") & (adata.obsm["X_umap"][:, 1] < 5) & (adata.obsm["X_umap"][:, 0] < 10)) != True]
adata = adata[((adata.obs["batch"] == "P4") & (adata.obsm["X_umap"][:, 1] < 50) & (adata.obsm["X_umap"][:, 0] < 3)) != True]

# print(min(adata.obsm["X_umap"][(adata.obs["batch"] == "P1"), 1]))

# exit()
# sc.pp.neighbors(adata, metric="cosine", use_rep="X")
# sc.tl.umap(adata)
# adata.write_h5ad(
#     "tmp_cite_full.h5ad"
# )

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
plt.subplots_adjust(wspace=0.6, left=0.05, right=0.9, bottom=0.08)

sc.pl.umap(adata, color="cell_type_l1", ax=axs[0], show=False)
sc.pl.umap(adata, color="batch", ax=axs[1], show=False)
# concat = adata.obsm["GEX_X_umap"]
# plt.plot(
#     concat[:, 0],
#     concat[:, 1],
#     color="gray",
#     linestyle="dashed",
#     linewidth=0.5,
# )
# plt.tight_layout()

plt.savefig("ex.png")
plt.savefig("ex.svg")

