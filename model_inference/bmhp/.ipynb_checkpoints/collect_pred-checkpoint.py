import os, shutil
import pandas as pd
import numpy as np

from scipy.stats import gaussian_kde

def estimate_MAP(data):
    kde = gaussian_kde(data)
    no_samples = 100 # number of interpolation points
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]

    return maxima
    
def process_preds(file, title="pred"):

    df = pd.read_csv(file).head(1000)
    
    nums = list(set([int(x.split("[")[1].split("]")[0]) for x in df.columns]))
    nums.sort()
    
    pred_df = df[[x for x in df.columns if title in x]]
    actual_df = df[[x for x in df.columns if 'actual' in x]]
    
    rel_df = pred_df.describe().T
    
    map_preds = []
    for pred_col in pred_df.columns:
        map_preds.append(estimate_MAP(pred_df[pred_col]))
    
    rel_df_summary = rel_df[["mean", "50%"]]
    rel_df_summary["mode"] = map_preds
    rel_df_summary["actual"] = actual_df.iloc[0,:].values
    rel_df_summary.columns = ["mean", "median", "mode", "actual"]
    rel_df_summary.index = [x.split("[")[1].split("]")[0] for x in rel_df_summary.index]
    
    return rel_df_summary

for dir1 in os.listdir("preds/"): # category
    dir1_ = "preds/"+dir1+"/"
    for dir2 in os.listdir(dir1_): # publisher
        dir2_ = dir1_ + dir2 + "/"
        for dir3 in os.listdir(dir2_): # model type
            dir3_ = dir2_ + dir3
            print(dir1, dir1_, dir2, dir2_, dir3, dir3_)
            try:
                df = process_preds(dir3_+ "/preds.csv", "pred")
                df.to_csv(f"preds_mean/{dir1}_{dir2}_{dir3}.csv")
            except:
                continue
