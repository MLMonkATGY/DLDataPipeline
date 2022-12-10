from pprint import pprint
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


def get_view_names():
    return [
        "front_view_left",
        "front_view",
        "front_view_right",
        "rear_view",
        "rear_view_left",
        "rear_view_right",
    ]


def get_cv_pred(expId):
    views = get_view_names()
    outputDir = pathlib.Path(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results"
    )
    os.makedirs(outputDir, exist_ok=True)
    for view in tqdm(views, desc="angle"):
        runName = f"cv_pred_{view}"
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


def combine_df():
    search = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results/cv_pred_**.csv"
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


def get_raw_multilabel_df():
    wr.config.s3_endpoint_url = "http://192.168.1.7:8333"
    srcBucketName = "multilabel_df"
    labelDf = wr.s3.read_parquet(path=f"s3://{srcBucketName}/", dataset=True)
    return labelDf


def ensemble_pred(completeDf: pd.DataFrame):
    allParts = completeDf["parts"].unique().tolist()
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

    return accDf, partPredDf, allParts


def eval_by_parts(allParts, partPerfDf):
    partMetrics = []
    old_err_state = np.seterr(divide='raise')
    np.seterr(divide='ignore')

    for part in allParts:
        gtCol = partPerfDf[f"gt_{part}"].values
        predCol = partPerfDf[f"pred_{part}"].values
        # posGt = gtCol[gtCol == 1]
        posGtCount = np.count_nonzero(gtCol == 1)
        negGtCount = np.count_nonzero(gtCol == 0)

        tp = np.count_nonzero(predCol[gtCol == 1] == 1) / posGtCount
        tn = np.count_nonzero(predCol[gtCol == 0] == 0) / negGtCount
        fp = np.count_nonzero(predCol[gtCol == 0] == 1) / negGtCount
        fn = np.count_nonzero(predCol[gtCol == 1] == 0) / posGtCount
        assert 0.99 < tp + fn < 1.01
        assert 0.99 < tn + fp < 1.01
        precision = np.divide(tp, (tp + fp))
        recall = np.divide(tp , (tp + fn))
        acc = (tp + tn) / (tp + tn + fp + fn)
        partMetrics.append(
            {
                "part": part,
                "tp": tp,
                "tn": tn,
                "fn": fn,
                "fp": fp,
                "precision": precision,
                "recall": recall,
                "acc": acc,
                "gt_pos_count": posGtCount,
                "gt_neg_count": negGtCount,
                "gt_pos_ratio": posGtCount / (posGtCount + negGtCount),
            }
        )

    partEvalMetrics = pd.json_normalize(partMetrics)
    partEvalMetrics.to_csv(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results/part_metrics.csv"
    )
    avgTp = partEvalMetrics["tp"].mean()
    avgTn = partEvalMetrics["tn"].mean()
    avgAcc = partEvalMetrics["acc"].mean()
    minTp = partEvalMetrics["tp"].min()
    minTn = partEvalMetrics["tn"].min()
    np.seterr(**old_err_state)
    return {
        "avgTp" : avgTp,
        "avgTn" : avgTn,
        "avgAcc" : avgAcc,
        "minTp" : minTp,
        "minTn" : minTn,

    }


if __name__ == "__main__":
    expId = 84
    get_cv_pred(expId)
    completePredDf = combine_df()
    ensemble_pred(completePredDf)
    