from pprint import pprint
import mlflow
import ujson as json
from mlflow.tracking import MlflowClient
from analysis.ensemble_predictions import get_raw_multilabel_df
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
import optuna


def combine_df(srcDir):
    search = f"{srcDir}/cv_pred_**.csv"
    allPredfile = glob.glob(search, recursive=True)
    allDf = []
    for p in allPredfile:
        viewName = p.split("/")[-1].split(".")[0].replace("cv_pred_", "")
        df = pd.read_csv(p)
        df["view"] = viewName
        allDf.append(df)
    completeDf = pd.concat(allDf)
    completeDf.dropna(subset="file", inplace=True)
    return completeDf


srcDir = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results"
completeDf = combine_df(srcDir)


def ensemble_pred_optimize(trial):
    x = trial.suggest_float("x", -10, 10)

    allParts = completeDf["parts"].unique().tolist()
    allViews = completeDf["view"].unique().tolist()
    completeDf["CaseID"] = completeDf["file"].apply(
        lambda x: int(x.split("/")[-1].split("_")[0])
    )
    caseIdList = completeDf["CaseID"].unique().tolist()
    labelDf = get_raw_multilabel_df()
    labelDf = labelDf[labelDf["CaseID"].isin(caseIdList)]
    allSubsetAcc = []
    logInterval = 1000
    partDmgPred = {f"{y}{x}": [] for x in allParts for y in ["gt_", "pred_"]}
    caseSubsetAcc = []
    # pprint(partDmgPred)
    for caseId in tqdm(caseIdList):
        casePartRows = completeDf[completeDf["CaseID"] == caseId]
        correct = 0
        for part in allParts:
            partPreds = casePartRows[casePartRows["parts"] == part]
            if partPreds.empty:
                gtLabel = labelDf[labelDf["CaseID"] == caseId][part].item()
                # No detection at all
                allPred = [0]
            else:
                gtLabel = partPreds["gt"].iloc[0]
                allPred = [int(x) for x in partPreds["pred"].tolist()]

            rankPreds = Counter(allPred).most_common(2)
            predDmgStatus = 0
            if len(rankPreds) > 1 and rankPreds[0][1] == rankPreds[1][1]:
                # Even num pred
                predDmgStatus = 1
            else:
                predDmgStatus = rankPreds[0][0]
            partDmgPred[f"gt_{part}"].append(gtLabel)
            partDmgPred[f"pred_{part}"].append(predDmgStatus)

            if predDmgStatus == gtLabel:
                correct += 1

        subsetAcc = correct / len(allParts)
        caseSubsetAcc.append({"CaseID": caseId, "subset_acc": subsetAcc})
        allSubsetAcc.append(subsetAcc)
        if len(allSubsetAcc) % logInterval == 0:
            tqdm.write(np.format_float_positional(np.mean(allSubsetAcc), 3))
    accDf = pd.json_normalize(caseSubsetAcc)
    partPredDf = pd.DataFrame(partDmgPred)
    accDf.to_csv(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results/acc_perf.csv"
    )
    partPredDf.to_csv(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results/part_perf.csv"
    )

    return accDf, partPredDf


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":

    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(
        objective,
        n_trials=100,
        n_jobs=5,
    )
