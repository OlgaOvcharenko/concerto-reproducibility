import argparse
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd


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
                        help='Adata path')

    args = parser.parse_args()
    return args


def evaluate_model(adata, batch_key="batch", cell_type_label="cell_type_l1"):
    names_obs = list(adata.obsm.keys())
    names_obs.remove("X_pca")
    names_obs.remove("Unintegrated_HVG_only")
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

def scale_result(df):
    cols = ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']
    df[cols] = MinMaxScaler().fit_transform(df[cols])
    scaled= pd.DataFrame(df, columns=df.columns, index=df.index)
    
    biometrics = [i for i in ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI']]
    batchmetrics = [i for i in ['Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']]
    scaled[f"Batch correction final"] = scaled[batchmetrics].mean(1)
    scaled[f"Bio conservation final"] = scaled[biometrics].mean(1)
    scaled[f"Total final"] = 0.6 * scaled[f"Bio conservation final"] + 0.4 * scaled[f"Batch correction final"]
    df[f"Bio conservation final"] = scaled[f"Bio conservation final"].copy()
    df[f"Batch correction final"] = scaled[f"Batch correction final"].copy()
    df[f"Total final"] = scaled[f"Total final"].copy()
    return df

args = get_args()
data = args.data
repeat = 0

only_files = [f'Multimodal_pretraining/data/{data}/' + f for f in listdir(f'./Multimodal_pretraining/data/{data}/') if isfile(join(f'./Multimodal_pretraining/data/{data}/', f)) if f.startswith(f'{data}_bc_')]

#  './Multimodal_pretraining/data/{data}/{data}_bc_{combine_omics}_mt_{model_type}_bs_{batch_size}_{epoch}_{lr}_{drop_rate}_0_1_{heads}_{repeat}.h5ad'

final_df = pd.DataFrame(columns=['combine_omics', 'model_type', 'batch_size', 'epoch', 'lr', 'drop_rate', 'heads', 'Embedding', 'Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison', 'Batch correction', 'Bio conservation', 'Total'])
for file_read in only_files:
    print(file_read)
    params = file_read.split("_bc_")[1].split("_")

    combine_omics = params[0]
    model_type = params[2]
    batch_size = params[4]
    epoch = params[5]
    lr = params[6]
    drop_rate = params[7]
    heads = params[10]

    adata = sc.read_h5ad(file_read) 
    print("Read adata")

    df = evaluate_model(adata=adata)
    df = df.assign(**{"combine_omics": combine_omics, "model_type": model_type, "batch_size": batch_size, "epoch": epoch, "lr": lr, "drop_rate": drop_rate, "heads": heads})
    # df = df.assign(**{"model_type": model_type})
    # df = df.assign(**{"batch_size": batch_size})
    # df = df.assign(**{"epoch": epoch})
    # df = df.assign(**{"lr": lr})
    # df = df.assign(**{"drop_rate": drop_rate})
    # df = df.assign(**{"heads": heads})
    print(df.columns)

    final_df = pd.concat([df, final_df], ignore_index=True)

final_df.to_csv(f'./Multimodal_pretraining/results/{data}/{data}_metrics_unscaled.csv')

final_df = scale_result(final_df)
final_df.to_csv(f'./Multimodal_pretraining/results/{data}/{data}_metrics_scaled.csv')
