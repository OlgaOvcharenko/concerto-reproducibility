import argparse
import sys
import pandas as pd
sys.path.append("../")
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm, gridspec, pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import umap


l2tol1 = {
    'CD8 Naive': 'CD8 T',
    'CD8 Proliferating': 'CD8 T',
    'CD8 TCM': 'CD8 T',
    'CD8 TEM': 'CD8 T',
    'CD8+ T': 'CD8 T',
    'CD8+ T naive': 'CD8 T',
    'CD4 CTL': 'CD4 T',
    'CD4 Naive': 'CD4 T',
    'CD4 Proliferating': 'CD4 T',
    'CD4 TCM': 'CD4 T',
    'CD4 TEM': 'CD4 T',
    'CD4+ T activated': 'CD4 T',
    'CD4+ T naive': 'CD4 T',
    'CD14+ Mono': 'CD14+ Mono',
    'CD16+ Mono': 'CD14+ Mono',
    'Treg': 'CD4 T',
    'NK': 'NK',
    'NK Proliferating': 'NK',
    'NK_CD56bright': 'NK',
    'dnT': 'other T',
    'gdT': 'other T',
    'ILC': 'other T',
    'MAIT': 'other T',
    'CD14 Mono': 'Monocytes',
    'CD16 Mono': 'Monocytes',
    'cDC1': 'DC',
    'cDC2': 'DC',
    'pDC': 'DC',
    'ASDC':'DC',
    'B intermediate': 'B',
    'B memory': 'B',
    'B naive': 'B',
    'B1 B': 'B',
    'Plasmablast': 'B',
    'Plasma cell': 'B',
    'Transitional B': 'B',
    'Naive CD20+ B': 'B',
    'Eryth': 'other',
    'HSPC': 'other',
    'Platelet': 'other',
    'Erythroblast': 'Erythroblast',
    'Proerythroblast': 'Proerythroblast',
    'Normoblast': 'Normoblast',
    'Lymph prog': 'Lymph prog',
    'G/M prog': 'G/M prog',
    'G/M prog': 'G/M prog',
    'HSC': 'HSC',
    'MK/E prog'          : 'MK/E prog',           
    'ILC'                : 'ILC',                 
    'ID2-hi myeloid prog': 'ID2-hi myeloid prog' 
}

def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Dataset (Simulated/Not)')

    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--lr', type= float, required=True,
                        help='learning rate')
    parser.add_argument('--batch_size', type= int, required=True,
                        help='batch size')
    parser.add_argument('--drop_rate', type= float, required=True,
                        help='dropout rate')
    parser.add_argument('--heads', type= int, required=True,
                        help='heads for NNs')
    parser.add_argument('--attention_t', type= int, required=True,
                        help='to use attention with teacher')
    parser.add_argument('--attention_s', type= int, required=True,
                        help='to use attention with student')
    parser.add_argument('--train', type= int, required=True,
                        help='to train or just inference')
    parser.add_argument('--test', type= int, required=True,
                        help='inference')
    parser.add_argument('--model_type', type= int, required=True,
                        help='1 for simple TT, else 4 together')
    parser.add_argument('--combine_omics', type= int, required=True,
                        help='0/1')

    args = parser.parse_args()
    return args

def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(
        0,
        0.5 *
        height,
        width,
        0,
        length_includes_head=True,
        head_width=0.75 *
        height)
    return p

def plot_train_only(ax, X_umap, y_train, name, labels, colormap="Paired"):
    colormap = matplotlib.colormaps[colormap].colors
    if len(labels) < 11:
        colormap = matplotlib.colormaps["Paired"].colors
    if len(labels) > 21:
        colormap = list(matplotlib.colormaps["tab20"].colors) 
        colormap.extend(list(matplotlib.colormaps["tab20b"].colors))
        
    scs = []
    for lbl in labels:
        sc = ax.scatter(
            X_umap[y_train==lbl, 0],
            X_umap[y_train==lbl, 1],
            marker=".",
            color=colormap[lbl],
            alpha=0.8,
            s=0.7
        )

        scs.append(sc)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title(name, size=8)
    # ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    # ax.xaxis.set_label_coords(x=0.5, y=-0.02)

    return scs, labels

def encode_all_y(y_train, y_test, y_predict):
    y_train = y_train.to_frame()
    y_train["cell_type_l1"] = y_train["cell_type_l1"].astype('str')

    y_test = y_test.to_frame()
    y_test["cell_type_l1"] = y_test["cell_type_l1"].astype('str')

    y_predict = y_predict.to_frame()
    y_predict.rename(columns={y_predict.columns[0]: "cell_type_l1"}, inplace=True)
    y_predict["cell_type_l1"] = y_predict["cell_type_l1"].astype('str')

    uniques = set()
    uniques.update(y_train["cell_type_l1"].unique().tolist()) 
    uniques.update(y_test["cell_type_l1"].unique().tolist())
    uniques.update(y_predict["cell_type_l1"].unique().tolist())

    label_types = dict()
    for i, lbl in enumerate(uniques):
        label_types[i] = lbl
        y_train.loc[y_train["cell_type_l1"]==lbl, "cell_type_l1"] = i
        y_test.loc[y_test["cell_type_l1"]==lbl, "cell_type_l1"] = i
        y_predict.loc[y_predict["cell_type_l1"]==lbl, "cell_type_l1"] = i

    return y_train, y_test, y_predict, label_types

def encode_all_batch(y_train, y_test):
    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    y_train["batch"] = y_train["batch"].astype('str')
    y_test["batch"] = y_test["batch"].astype('str')

    uniques = set()
    uniques.update(y_train["batch"].unique().tolist()) 
    uniques.update(y_test["batch"].unique().tolist())

    label_types = dict()
    for i, lbl in enumerate(uniques):
        label_types[lbl] = i
        y_train.loc[y_train["batch"]==lbl, "batch"] = i
        y_test.loc[y_test["batch"]==lbl, "batch"] = i

    return y_train, y_test, label_types

def encode_all_leiden(y_train):
    y_train = y_train.to_frame()
    y_train.rename(columns={list(y_train.columns)[0]: "leiden"}, inplace=True)
    y_train["leiden"] = y_train["leiden"].astype('str')

    uniques = set()
    uniques.update(y_train["leiden"].unique().tolist()) 

    label_types = dict()
    for i, lbl in enumerate(uniques):
        label_types[lbl] = i
        y_train.loc[y_train["leiden"]==lbl, "leiden"] = i

    return y_train, label_types

def get_joint_umap(X_train, X_test):
    reducer = umap.UMAP()
    return reducer.fit_transform(np.append(X_train, X_test, axis=0))

def get_umap(X):
    reducer = umap.UMAP()
    return reducer.fit_transform(X)

def create_plot_qr(adata_merged_train, adata_merged_test, 
                data: str, attention_t: bool, attention_s: bool,
                batch_size:int, epoch: int, lr: float, drop_rate: float, 
                heads: int, combine_omics: int, model_type: int, only_RNA: bool):
    X_train = adata_merged_train.obsm[f'train_{epoch}_encoder_{drop_rate}_{only_RNA}']
    X_test = adata_merged_test.obsm[f'test_{epoch}_encoder_{drop_rate}_{only_RNA}_0']

    X_train_umap = adata_merged_train.obsm[f'train_umap_{epoch}_encoder_{drop_rate}_{only_RNA}']
    X_train_leiden = adata_merged_train.obs[f'train_leiden_{epoch}_encoder_{drop_rate}_{only_RNA}']
    X_test_umap = adata_merged_test.obsm[f'test_umap_{epoch}_encoder_{drop_rate}_{only_RNA}_0']

    y_train = adata_merged_train.obs['cell_type_l1']
    y_test = adata_merged_test.obs['cell_type_l1']
    y_pred = adata_merged_test.obs[f'pred_cell_type_{epoch}_encoder_{drop_rate}_{only_RNA}_0']
    y_train, y_test, y_pred, label_types = encode_all_y(y_train=y_train, y_test=y_test, y_predict=y_pred)

    # y_train_batch = adata_merged_train.obs['batch']
    # y_test_batch = adata_merged_test.obs['batch']
    # y_train_batch, y_test_batch, batch_types = encode_all_batch(y_train=y_train_batch, y_test=y_test_batch)

    fig = plt.figure(
        constrained_layout=True,
        figsize=(7.125, ((7.125 - 0.17) / 1.65) / 1.6),
        dpi=1000,
        facecolor="w",
        edgecolor="k",
    )
    spec2 = gridspec.GridSpec(
        ncols=3, nrows=1, figure=fig, left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.05
    )
    ax00 = fig.add_subplot(spec2[0, 0])
    ax01 = fig.add_subplot(spec2[0, 1])
    ax02 = fig.add_subplot(spec2[0, 2])

    # Train only
    sc00, sc00_labels = plot_train_only(ax=ax00, X_umap=X_train_umap, y_train=y_train["cell_type_l1"], name="Reference", labels=y_train["cell_type_l1"].unique().tolist(), colormap="tab20b")

    # # Train + test
    # X_joint_umap = get_joint_umap(X_train, X_test)
    # ax01.scatter(X_joint_umap[0:X_train.shape[0], 0], X_joint_umap[0:X_train.shape[0], 1], marker=".", c="gray", alpha=0.3, s=1)
    # sc01, sc01_labels = plot_train_only(ax=ax01, X_umap=X_joint_umap[X_train.shape[0]:, :], y_train=y_test["cell_type_l1"], name="Query True Labels", labels=y_train["cell_type_l1"].unique().tolist(), colormap="tab20b")

    # # Train + pred
    # ax02.scatter(X_joint_umap[0:X_train.shape[0], 0], X_joint_umap[0:X_train.shape[0], 1], marker=".", c="gray", alpha=0.3, s=1)
    # sc02, sc02_labels = plot_train_only(ax=ax02, X_umap=X_joint_umap[X_train.shape[0]:, :], y_train=y_pred["cell_type_l1"], name="Query Prediction", labels=y_pred["cell_type_l1"].unique().tolist(), colormap="tab20b")

    # Train + test
    X_joint_umap = get_joint_umap(X_train, X_test)
    # print(X_test_umap.shape)
    print(y_test["cell_type_l1"].shape)
    sc01, sc01_labels = plot_train_only(ax=ax01, X_umap=X_test_umap, y_train=y_test["cell_type_l1"], name="Query True Labels", labels=y_train["cell_type_l1"].unique().tolist(), colormap="tab20b")

    # Train + pred
    # print(X_test_umap.shape)
    print(y_pred["cell_type_l1"].shape)
    sc02, sc02_labels = plot_train_only(ax=ax02, X_umap=X_test_umap, y_train=y_pred["cell_type_l1"], name="Query Prediction", labels=y_pred["cell_type_l1"].unique().tolist(), colormap="tab20b")


    # # Batch train + test
    # y_merged = np.append(y_train_batch, y_test_batch, axis=0)
    # y_merged = y_merged.reshape((y_merged.shape[0],))
    # sc10_batch, sc10_batch_labels = plot_train_only(ax=ax10, X_umap=X_joint_umap, y_train=y_merged, name="Batch Train + Test", labels=np.unique(y_merged), colormap="tab20b")

    # # Test only
    # sc11, sc11_labels = plot_train_only(ax=ax11, X_umap=X_test_umap, y_train=y_test["cell_type_l1"], name="Test", labels=y_test["cell_type_l1"].unique().tolist())

    # # Predict only
    # sc12, sc12_labels = plot_train_only(ax=ax12, X_umap=X_test_umap, y_train=y_pred["cell_type_l1"], name="Predict", labels=y_pred["cell_type_l1"].unique().tolist())

    arrows_dict = {}
    for i, val in enumerate(sc00_labels):
        arrows_dict[label_types[val]] = sc00[i]
    for i, val in enumerate(sc01_labels):
        arrows_dict[label_types[val]] = sc01[i]
    for i, val in enumerate(sc02_labels):
        arrows_dict[label_types[val]] = sc02[i]
    # for i, val in enumerate(sc11_labels):
    #     arrows_dict[label_types[val]] = sc11[i]
    # for i, val in enumerate(sc12_labels):
    #     arrows_dict[label_types[val]] = sc12[i]

    dict_vals = list(arrows_dict.values())
    dict_keys = list(arrows_dict.keys())
    
    leg = fig.legend(
        dict_vals,
        dict_keys,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=7,
        ncol=6,
        markerscale=10,
    )

    save_fig = f'qr_{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{only_RNA}.pdf'
    fig.savefig(f"plots_final/{data}/{save_fig}", bbox_extra_artists=(leg,), bbox_inches='tight')
    
def create_plot_bc(adata_merged_train, adata_merged_test, 
                data: str, attention_t: bool, attention_s: bool,
                batch_size:int, epoch: int, lr: float, drop_rate: float, 
                heads: int, combine_omics: int, model_type: int, only_RNA: bool):
    X_train = adata_merged_train.obsm[f'train_{epoch}_encoder_{drop_rate}_{only_RNA}']
    X_test = adata_merged_test.obsm[f'test_{epoch}_encoder_{drop_rate}_{only_RNA}_0']

    X_train_umap = adata_merged_train.obsm[f'train_umap_{epoch}_encoder_{drop_rate}_{only_RNA}']
    X_train_leiden = adata_merged_train.obs[f'train_leiden_{epoch}_encoder_{drop_rate}_{only_RNA}']
    X_test_umap = adata_merged_test.obsm[f'test_umap_{epoch}_encoder_{drop_rate}_{only_RNA}_0']

    y_train = adata_merged_train.obs['cell_type_l1']
    y_test = adata_merged_test.obs['cell_type_l1']
    y_pred = adata_merged_test.obs[f'pred_cell_type_{epoch}_encoder_{drop_rate}_{only_RNA}_0']
    y_train, y_test, y_pred, label_types = encode_all_y(y_train=y_train, y_test=y_test, y_predict=y_pred)

    y_train_batch = adata_merged_train.obs['batch']
    y_test_batch = adata_merged_test.obs['batch']
    y_train_batch, y_test_batch, batch_types = encode_all_batch(y_train=y_train_batch, y_test=y_test_batch)

    X_train_leiden, leiden_types = encode_all_leiden(X_train_leiden)

    fig = plt.figure(
        constrained_layout=True,
        figsize=(7.125, ((7.125 - 0.17) / 1.65) / 1.6),
        dpi=1000,
        facecolor="w",
        edgecolor="k",
    )
    spec2 = gridspec.GridSpec(
        ncols=3, nrows=1, figure=fig, left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.05
    )
    ax00 = fig.add_subplot(spec2[0, 0])
    ax01 = fig.add_subplot(spec2[0, 1])
    ax02 = fig.add_subplot(spec2[0, 2])

    # Train only
    sc00, sc00_labels = plot_train_only(ax=ax00, X_umap=X_train_umap, y_train=y_train["cell_type_l1"], name="Train", labels=y_train["cell_type_l1"].unique().tolist(), colormap="tab20b")
    
    # Batch train + test
    sc01_batch, sc01_batch_labels = plot_train_only(ax=ax01, X_umap=X_train_umap, y_train=y_train_batch["batch"], name="Batch Id", labels=np.unique(y_train_batch["batch"]), colormap="tab20b")

    # Test only
    sc11, sc11_labels = plot_train_only(ax=ax02, X_umap=X_train_umap, y_train=X_train_leiden["leiden"], name="Leiden", labels=X_train_leiden["leiden"].unique().tolist(), colormap="tab20")

    arrows_dict = {}
    for i, val in enumerate(sc00_labels):
        arrows_dict[label_types[val]] = sc00[i]
    # for i, val in enumerate(sc01_batch_labels):
    #     arrows_dict[label_types[val]] = sc01_batch[i]
    # for i, val in enumerate(sc11_labels):
    #     arrows_dict[label_types[val]] = sc11[i]

    dict_vals = list(arrows_dict.values())
    dict_keys = list(arrows_dict.keys())
    
    leg = fig.legend(
        dict_vals,
        dict_keys,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=7,
        ncol=6,
        markerscale=10,
    )

    save_fig = f'bc_{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}_{only_RNA}.pdf'
    fig.savefig(f"plots_final/{data}/{save_fig}", bbox_extra_artists=(leg,), bbox_inches='tight')

def main():
    # Parse args
    args = get_args()
    data = args.data
    epoch = args.epoch
    lr = args.lr
    batch_size= args.batch_size
    drop_rate= args.drop_rate
    attention_t = True if args.attention_t == 1 else False
    attention_s = True if args.attention_s == 1 else False 
    heads = args.heads
    train = args.train 
    model_type = args.model_type
    test = args.test
    combine_omics = args.combine_omics

    print(f"Plot: epoch {epoch}, model type {model_type}, lr {lr}, batch_size {batch_size}, drop_rate {drop_rate}, attention_t {attention_t}, attention_s {attention_s}, heads {heads}.")
    
    # Read data
    # filename_train = f'./Multimodal_pretraining/data/{data}/{data}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    # filename_test = f'./Multimodal_pretraining/data/{data}/{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'

    filename_train = f'./Multimodal_pretraining/data/{data}_train_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
    filename_test = f'./Multimodal_pretraining/data/{data}_test_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'


    adata_merged_train = sc.read_h5ad(filename_train)
    adata_merged_test = sc.read_h5ad(filename_test)

    ep_vals = [4]
    i = 4
    while i < epoch:
        ep_vals.append(i)
        i = i * 2
    ep_vals.append(epoch)

    only_RNAs = [True, False] if combine_omics == 0 else [False]
    for only_RNA in only_RNAs:
        for dr in [0.0]:
                for e in [64, epoch]: 
                    # print(adata_merged_train)
                    # print(adata_merged_test)
                    
                    create_plot_bc(adata_merged_train=adata_merged_train, 
                                adata_merged_test=adata_merged_test, 
                                data=data, attention_t=attention_t, attention_s=attention_s,
                                batch_size=batch_size, epoch=e, lr=lr, drop_rate=dr, 
                                heads=heads, combine_omics=combine_omics, model_type=model_type,
                                only_RNA=only_RNA)
                    

                    create_plot_qr(adata_merged_train=adata_merged_train, 
                                adata_merged_test=adata_merged_test, 
                                data=data, attention_t=attention_t, attention_s=attention_s,
                                batch_size=batch_size, epoch=e, lr=lr, drop_rate=dr, 
                                heads=heads, combine_omics=combine_omics, model_type=model_type,
                                only_RNA=only_RNA)

main()
