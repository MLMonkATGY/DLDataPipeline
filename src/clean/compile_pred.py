from pprint import pprint
from typing import List
import mlflow
import ujson as json
from mlflow.tracking import MlflowClient
from data import ImportEnv
import pathlib
import os
from tqdm import tqdm
import glob
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader2
import torch
from torch.cuda.amp import autocast
import torchmetrics
import numpy as np
import itertools
import ast
from joblib import Parallel, delayed
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    confusion_matrix,
)
from collections import Counter
import awswrangler as wr
from src.TrainClassifierParams import trainParams
import warnings
from cleanlab.filter import find_label_issues
from cleanlab.internal.multilabel_utils import onehot2int


def get_view_names():
    return [
        "front_view_left",
        "front_view",
        "front_view_right",
        "rear_view",
        "rear_view_left",
        "rear_view_right",
    ]


def get_cv_pred(expId, vehicleType, downloadDir, view):
    outputDir = pathlib.Path(downloadDir)
    os.makedirs(outputDir, exist_ok=True)
    runName = f"cv_pred_{vehicleType}_{view}"
    query = f"tags.`mlflow.runName`='{runName}'"
    runs = MlflowClient().search_runs(
        experiment_ids=[expId],
        filter_string=query,
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    info = runs[0].info
    runId = info.run_id
    mlflow.artifacts.download_artifacts(run_id=runId, dst_path=outputDir)


def combine_df(downloadDir, view):
    search = f"{downloadDir}/cv_pred_**{view}.csv"
    allPredfile = glob.glob(search, recursive=False)
    allDf = []
    for p in allPredfile:
        viewName = p.split("/")[-1].split(".")[0].replace("cv_pred_", "")
        print(viewName)
        df = pd.read_csv(p)
        df["view"] = viewName
        allDf.append(df)
    completeDf = pd.concat(allDf)
    completeDf.dropna(subset="file", inplace=True)
    return completeDf


def compile_pred_proba(completeDf: pd.DataFrame, allParts: List[str], valuesCol: str):
    completeDf["CaseID"] = completeDf["file"].apply(
        lambda x: int(x.split("/")[-1].split("_")[0])
    )
    df_wide = completeDf.pivot_table(index="CaseID", columns="part", values=valuesCol)
    print(df_wide)
    return df_wide
    print(df_wide)


if __name__ == "__main__":
    downloadDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/cleanlab"
    expId = 108
    vehicleType = "Saloon - 4 Dr"
    views = get_view_names()
    for view in views:
        get_cv_pred(expId, vehicleType, downloadDir, view)
        completePredDf = combine_df(downloadDir, view)
        allParts = completePredDf["part"].unique().tolist()
        viewPredDf = compile_pred_proba(completePredDf, allParts, "conf")
        viewGtDf = compile_pred_proba(completePredDf, allParts, "gt")
        labels = onehot2int(viewGtDf.values)
        pred_labels = onehot2int(np.array((viewPredDf.values > 0.5), np.int32))
        ranked_label_issues = find_label_issues(
            labels=labels,
            pred_probs=viewPredDf.values,
            multi_label=True,
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
            filter_by="confident_learning",
        )
        print(ranked_label_issues)
