import argparse
import tempfile

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import muon
import scanpy as sc
import scvi
import seaborn as sns
import torch

import os
import sys

import pandas as pd
sys.path.append("../")
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

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
    parser.add_argument('--task', type= int, required=True,
                        help='0-bc, 1-qr+mp')

    args = parser.parse_args()
    return args

def prepare_data_PBMC_together(train: bool = True, save_path: str = ''):
    print("Read PBMC data.")
    adata_RNA = sc.read_h5ad(save_path + f'adata_RNA_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_Protein_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_RNA_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_Protein_test.h5ad')

    print(f"RNA data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"Protein data shape {adata_Protein.shape}, test {adata_Protein_test.shape}")

    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    adata_merged_test = ad.concat([adata_RNA_test, adata_Protein_test], axis=1)
    sc.tl.pca(adata_merged_test)
    adata_merged_test.obsm["Unintegrated_HVG_only"] = adata_merged_test.obsm["X_pca"]
    
    adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test

def prepare_data_PBMC_full(train: bool = True, save_path: str = ''):
    print("Read PBMC data.")
    adata_RNA = sc.read_h5ad(save_path + f'adata_RNA_full.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_Protein_full.h5ad')

    print(f"RNA data shape train {adata_RNA.shape}")
    print(f"Protein data shape {adata_Protein.shape}")

    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]
    
    print("Preprocessed data.")
    return adata_merged, adata_RNA, adata_Protein

def prepare_data_neurips_cite_full(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_neurips_GEX_full.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_neurips_ADT_full.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}")
    print(f"ADT data shape train {adata_Protein.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    return adata_merged, adata_RNA, adata_Protein

def prepare_data_neurips_cite_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_GEX_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_ADT_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_GEX_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_ADT_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1)
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    adata_merged_test = ad.concat([adata_RNA_test, adata_Protein_test], axis=1)
    sc.tl.pca(adata_merged_test)
    adata_merged_test.obsm["Unintegrated_HVG_only"] = adata_merged_test.obsm["X_pca"]

    print("Saved adata.")
    return adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test


def read_data(data: str = "simulated", save_path: str = "", task=0):
    if data == "simulated":
        if task == 0:
            adata_merged, adata_RNA, adata_Protein = prepare_data_PBMC_full(train=True, save_path=save_path)
        else:
            adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test = prepare_data_PBMC_together(train=True, save_path=save_path)
    
    elif data == "human_cite":
        if task == 0:
            adata_merged, adata_RNA, adata_Protein = prepare_data_neurips_cite_full(train=True, save_path=save_path)
        else:
            adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test = prepare_data_neurips_cite_together(train=True, save_path=save_path)
    
    if task == 0:
        return adata_merged, adata_RNA, adata_Protein
    else:
        return adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test

def save_merged_adata(adata_merged, filename):
    adata_merged.write(filename)

    print(adata_merged)
    print(f"Saved adata all at {filename}")

def train_scvi(adata_RNA, adata_Protein):
    # Settings
    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    sc.set_figure_params(figsize=(6, 6), frameon=False)
    sns.set_theme()
    torch.set_float32_matmul_precision("high")
    save_dir = tempfile.TemporaryDirectory()

    sc.set_figure_params(figsize=(6, 6), frameon=False)
    sns.set_theme()
    torch.set_float32_matmul_precision("high")

    mdata = md.MuData({"rna": adata_RNA, "protein": adata_Protein})
    scvi.model.TOTALVI.setup_mudata(
        mdata,
        rna_layer="counts",
        protein_layer=None,
        batch_key="batch",
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein",
        },
    )

    model = scvi.model.TOTALVI(mdata)
    model.train()

    # arbitrarily store latent in rna modality
    rna = mdata.mod["rna_subset"]
    protein = mdata.mod["protein"]
    TOTALVI_LATENT_KEY = "X_totalVI"
    embedding = model.get_latent_representation()
    rna.obsm[TOTALVI_LATENT_KEY] = embedding
    return rna, embedding

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
    task = args.task

    print(f"sc-VI: epoch {epoch}, lr {lr}, batch_size {batch_size}, task {task}.")
    
    # Read data
    save_path = './Multimodal_pretraining/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    res_df = pd.DataFrame(columns=["accuracy", "f1_median", "f1_macro", "f1_weighted", "pearson" ])
    
    if task == 0:
        adata_merged, adata_RNA, adata_Protein = read_data(data=data, save_path=save_path, task=task)
    elif task == 1:
        adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test = read_data(data=data, save_path=save_path, task=task)
        adata_RNA_test.obs_names_make_unique()
        adata_Protein_test.obs_names_make_unique()

    adata_RNA.obs_names_make_unique()
    adata_Protein.obs_names_make_unique()

    # Train
    weight_path = save_path + 'weight/'
    if train:
        rna, embedding = train_scvi()
    print("Trained.")

    if test:
        if task == 0:
            filename = f'./Multimodal_pretraining/data/{data}/{data}_bc_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
            save_merged_adata(adata_merged=rna, filename=filename)

        else:
            pass
            # # Query-to-reference
            # # Test on train data
            # adata_merged = test_concerto_qr(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path, Protein_tf_path_test=Protein_tf_path, data=data, 
            #         attention_t=attention_t, attention_s=attention_s,
            #         batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
            #         heads=heads, combine_omics=combine_omics, model_type=model_type, 
            #         save_path=save_path, train=True, adata_merged=adata_merged, adata_RNA=adata_RNA, repeat=repeat)
            
            # filename = f'./Multimodal_pretraining/results/sc-vi_{data}_qr_train_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
            # save_merged_adata(adata_merged=adata_merged, filename=filename)

            # # Test on test data
            # adata_merged_test, acc, f1_median, f1_macro, f1_weighted = test_concerto_qr(weight_path=weight_path, RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, data=data, 
            #         attention_t=attention_t, attention_s=attention_s,
            #         batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
            #         heads=heads, combine_omics=combine_omics, model_type=model_type, 
            #         save_path=save_path, train=False, adata_merged=adata_merged_test, adata_RNA=adata_RNA_test, adata_merged_train=adata_merged, repeat=repeat)

            # filename = f'./Multimodal_pretraining/results/sc-vi_{data}_qr_test_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.h5ad'
            # save_merged_adata(adata_merged=adata_merged_test, filename=filename)

            # # Model prediction
            # pearson = test_concerto_mp(weight_path=weight_path, data=data, 
            #                     RNA_tf_path_test=RNA_tf_path_test, Protein_tf_path_test=Protein_tf_path_test, 
            #                     RNA_tf_path=RNA_tf_path, Protein_tf_path=Protein_tf_path, 
            #                     attention_t=attention_t, attention_s=attention_s,
            #                     batch_size=batch_size, epoch=epoch, lr=lr, drop_rate=drop_rate, 
            #                     heads=heads, combine_omics=combine_omics, model_type=model_type, 
            #                     save_path=save_path, repeat=repeat)
            
            # res_df.loc[repeat] = [acc, f1_median, f1_macro, f1_weighted, pearson]

    # if task != 0:
    #     res_df.to_csv(f'./Multimodal_pretraining/results/sc-vi_{data}_qr_train_{batch_size}_{epoch}_{lr}_{drop_rate}_{attention_s}_{attention_t}_{heads}.csv')

main()
