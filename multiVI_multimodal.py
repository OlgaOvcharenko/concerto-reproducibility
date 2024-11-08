import argparse

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import scanpy as sc
import scvi
import muon
import seaborn as sns
import torch

import os
import sys

import pandas as pd
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

_BIO_METRICS = BioConservation(isolated_labels=True, 
                               nmi_ari_cluster_labels_leiden=True, 
                               nmi_ari_cluster_labels_kmeans=False, 
                               silhouette_label=True, 
                               clisi_knn=True
                               )
_BATCH_METRICS = BatchCorrection(graph_connectivity=True, 
                                 kbet_per_label=True, 
                                 ilisi_knn=True, 
                                 pcr_comparison=True, 
                                 silhouette_batch=True
                                 )

def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Dataset (Simulated/Not)')

    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--train', type= int, required=True,
                        help='to train or just inference')
    parser.add_argument('--test', type= int, required=True,
                        help='inference')
    parser.add_argument('--task', type= int, required=True,
                        help='0-bc, 1-qr+mp')

    args = parser.parse_args()
    return args

def prepare_data_neurips_multiome_full(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_neurips_GEX_multiome_full.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_neurips_ATAC_multiome_full.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}")
    print(f"ADT data shape train {adata_Protein.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1, merge="same")
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    return adata_merged, adata_RNA, adata_Protein

def prepare_data_neurips_multiome_together(train: bool = True, save_path: str = ''):
    print("Read human data")
    adata_RNA = sc.read_h5ad(save_path + f'adata_GEX_multiome_train.h5ad')
    adata_Protein = sc.read_h5ad(save_path + f'adata_ATAC_multiome_train.h5ad')

    adata_RNA_test = sc.read_h5ad(save_path + f'adata_GEX_multiome_test.h5ad')
    adata_Protein_test = sc.read_h5ad(save_path + f'adata_ATAC_multiome_test.h5ad')

    print(f"GEX data shape train {adata_RNA.shape}, test {adata_RNA_test.shape}")
    print(f"ADT data shape train {adata_Protein.shape}, test {adata_Protein_test.shape}")

    # Add PCA after preprocessing for benchmarking
    adata_merged = ad.concat([adata_RNA, adata_Protein], axis=1, merge="same")
    sc.tl.pca(adata_merged)
    adata_merged.obsm["Unintegrated_HVG_only"] = adata_merged.obsm["X_pca"]

    adata_merged_test = ad.concat([adata_RNA_test, adata_Protein_test], axis=1, merge="same")
    sc.tl.pca(adata_merged_test)
    adata_merged_test.obsm["Unintegrated_HVG_only"] = adata_merged_test.obsm["X_pca"]

    print("Saved adata.")
    return adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test


def read_data(data: str = "human_multiome", save_path: str = "", task=0):
    if data == "human_multiome":
        if task == 0:
            adata_merged, adata_RNA, adata_Protein = prepare_data_neurips_multiome_full(train=True, save_path=save_path)
        else:
            adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test = prepare_data_neurips_multiome_together(train=True, save_path=save_path)
    
    if task == 0:
        return adata_merged, adata_RNA, adata_Protein
    else:
        return adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test

def save_merged_adata(adata_merged, filename):
    adata_merged.write(filename)

    print(adata_merged)
    print(f"Saved adata all at {filename}")

def train_scvi(adata_merged, adata_RNA, adata_atac):
    # Settings
    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    sc.set_figure_params(figsize=(6, 6), frameon=False)
    sns.set_theme()
    torch.set_float32_matmul_precision("high")

    sc.set_figure_params(figsize=(6, 6), frameon=False)
    sns.set_theme()
    torch.set_float32_matmul_precision("high")
    
    adata_mvi = scvi.data.organize_multiome_anndatas(adata_merged)
    scvi.model.MULTIVI.setup_anndata(adata_mvi)
    mvi = scvi.model.MULTIVI(
        adata_mvi,
        n_genes=(adata_mvi.var["feature_types"] == "GEX").sum(),
        n_regions=(adata_mvi.var["feature_types"] == "ATAC").sum(),
    )
    mvi.view_anndata_setup()
    mvi.train()
    embedding = mvi.get_latent_representation()
    adata_RNA.obsm["MultiVI_latent"] = embedding

    return adata_RNA, embedding

def evaluate_model(adata, batch_key="batch", cell_type_label="cell_type_l1"):
    names_obs = ['MultiVI_latent']
    print(names_obs)
    bm = Benchmarker(
                adata,
                batch_key=batch_key,
                label_key=cell_type_label,
                embedding_obsm_keys=names_obs,
                bio_conservation_metrics=_BIO_METRICS,
                batch_correction_metrics=_BATCH_METRICS,
                n_jobs=4,
            )
    bm.benchmark()
    a = bm.get_results(False, True)
    results = a.round(decimals=4)
    return results

def train_qr_scvi(adata_merged, adata_RNA, adata_Protein, adata_merged_test, adata_RNA_test, adata_Protein_test):
    # Settings
    scvi.settings.seed = 0

    adata_mvi = scvi.data.organize_multiome_anndatas(adata_merged)
    scvi.model.MULTIVI.setup_anndata(adata_mvi)

    adata_mvi_test = scvi.data.organize_multiome_anndatas(adata_merged_test)
    scvi.model.MULTIVI.setup_anndata(adata_mvi_test)

    mvi = scvi.model.MULTIVI(
        adata_mvi,
        n_genes=(adata_mvi.var["feature_types"] == "GEX").sum(),
        n_regions=(adata_mvi.var["feature_types"] == "ATAC").sum(),
    )
    mvi.view_anndata_setup()
    mvi.train()
    embedding = mvi.get_latent_representation()
    adata_RNA.obsm["MultiVI_latent"] = embedding

    # Query
    scvi.model.MULTIVI.prepare_query_anndata(adata_mvi_test, mvi)
    model_query = scvi.model.MULTIVI.load_query_data(adata_mvi_test, mvi)
    model_query.train(
        max_epochs=100,
        plan_kwargs=dict(weight_decay=0.0, scale_adversarial_loss=0.0),
    )
    
    embedding_test = model_query.get_latent_representation(adata_mvi_test)
    adata_RNA_test.obsm["X_totalVI_test"] = embedding_test

    # predict cell types of query
    print(adata_RNA_test)
    predictions = model_query.latent_space_classifer_.predict(adata_RNA_test.obsm["X_multivi_scarches"])
    categories = adata_RNA.obs["cell_type_l1"].astype("category").cat.categories
    cat_preds = [categories[i] for i in predictions]
    adata_RNA_test.obs["predicted_l2"] = cat_preds

    cell_types_list = pd.unique(adata_RNA_test.obs['cell_type_l1']).tolist()
    acc = accuracy_score(adata_RNA_test.obs['cell_type_l1'].to_list(), cat_preds)
    f1 = f1_score(adata_RNA_test.obs['cell_type_l1'].to_list(), cat_preds, labels=cell_types_list, average=None)
    f1_weighted = f1_score(adata_RNA_test.obs['cell_type_l1'].to_list(), cat_preds, labels=cell_types_list, average='weighted')
    f1_macro = f1_score(adata_RNA_test.obs['cell_type_l1'].to_list(), cat_preds, labels=cell_types_list, average='macro')
    f1_median = np.median(f1)
    
    print(f"Per class {cell_types_list} F1 {f1}")
    print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)


    return adata_RNA, embedding, adata_RNA_test, embedding_test

def main():
    # Parse args
    args = get_args()
    data = args.data
    epoch = args.epoch
    train = args.train 
    test = args.test
    task = args.task

    print(f"sc-VI: epoch {epoch}, task {task}.")
    
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
    adata_RNA.X = adata_RNA.X.toarray()
    adata_Protein.X = adata_Protein.X.toarray()
    adata_RNA.obs_names_make_unique()
    adata_Protein.obs_names_make_unique()

    # Train
    weight_path = save_path + 'weight/'
    if train:
        if task == 0:
            rna, embedding = train_scvi(adata_merged=adata_merged, adata_RNA=adata_RNA, adata_atac=adata_Protein)
        else:
            rna, embedding, rna_test, embedding_test = train_qr_scvi(adata_RNA=adata_RNA, adata_Protein=adata_Protein, adata_RNA_test=adata_RNA_test, adata_Protein_test=adata_Protein_test)
    print("Trained.")

    if test:
        if task == 0:
            filename = f'./Multimodal_pretraining/data/{data}/{data}_bs_{epoch}.h5ad'
            save_merged_adata(adata_merged=rna, filename=filename)
            final_df = evaluate_model(adata=rna)
            final_df.to_csv(f'./Multimodal_pretraining/results/{data}/{data}_totalvi_metrics_unscaled.csv')
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
