import argparse
from sklearn.preprocessing import MinMaxScaler
from os import listdir
from os.path import isfile, join
import pandas as pd

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


def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Adata path')

    args = parser.parse_args()
    return args


args = get_args()
data = args.data


df_old = pd.read_csv(data, sep=";")

df_old.loc[(df_old["combine_omics"] == 0) & (df_old["model_type"] == 1), "Embedding"] = "CLIP + Teacher" 
df_old.loc[(df_old["combine_omics"] == 0) & (df_old["model_type"] == 2), "Embedding"] = "CLIP + Teacher-Student Between Modalities" 
df_old.loc[(df_old["combine_omics"] == 0) & (df_old["model_type"] == 3), "Embedding"] = "CLIP + Teacher-Student All Pairs" 
df_old.loc[(df_old["combine_omics"] == 0) & (df_old["model_type"] == 4), "Embedding"] = "CLIP + NTXent Teacher" 
df_old.loc[(df_old["combine_omics"] == 0) & (df_old["model_type"] == 5), "Embedding"] = "CLIP + NTXent Teacher-Student" 
df_old.loc[(df_old["combine_omics"] == 1) & (df_old["model_type"] == 0), "Embedding"] = "Concerto" 

# print(df_old.loc[(df_old.Embedding == "CLIP + Teacher-Student Between Modalities") & (df_old.heads == 16)].to_dict())
# print(df_old.loc[(df_old.Embedding == "CLIP + Teacher-Student All Pairs") & (df_old.heads == 16)].to_dict())
# print(df_old.loc[(df_old.Embedding == "scCLIP")].to_dict())
# print(df_old.loc[(df_old.Embedding == "totalVI")].to_dict())
print(df_old.loc[((df_old.Embedding == "Concerto") & (df_old.epoch == 150))].to_dict())
# print(df_old.loc[((df_old.Embedding == "Concerto") & (df_old.heads == 128))].to_dict())
exit()

# df = scale_result(df_old).round(3).sort_values(by=['Total final'], ascending=False)

# print(df)
# df.to_csv(f'./Multimodal_pretraining/results/{data.split("/")[2]}/merged_scaled.csv')
# print(df.to_latex(index=False, columns=["Embedding", "epoch", "lr", "heads", "Bio conservation final", "Batch correction final", "Total final"]))
