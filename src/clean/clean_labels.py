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
from cleanlab.outlier import OutOfDistribution

from cleanlab.dataset import health_summary
from cleanlab.multilabel_classification import get_label_quality_scores

import torchvision
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
from skimage import io
from loguru import logger

wr.config.s3_endpoint_url = "http://192.168.1.4:8333"


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
    b = "cv_pred_SUV - 5 Dr_front_view"
    runs = MlflowClient().search_runs(
        experiment_ids=[expId],
        filter_string=query,
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    info = runs[0].info
    runId = info.run_id
    mlflow.artifacts.download_artifacts(run_id=runId, dst_path=outputDir)


def combine_df(downloadDir, filename):
    targetPathname = f"{downloadDir}/{filename}"

    viewName = targetPathname.split("/")[-1].split(".")[0].replace("cv_pred_", "")
    df = pd.read_csv(targetPathname)
    df["view"] = viewName

    df.dropna(subset="file", inplace=True)
    return df


def compile_pred_proba(completeDf: pd.DataFrame, allParts: List[str], valuesCol: str):
    completeDf["CaseID"] = completeDf["file"].apply(
        lambda x: int(x.split("/")[-1].split("_")[0])
    )
    df_wide = completeDf.pivot_table(index="CaseID", columns="part", values=valuesCol)
    caseFileDf = completeDf.groupby("CaseID").head(1)[["CaseID", "file", "view"]]
    assert len(caseFileDf) == len(df_wide)
    filesDf = df_wide.merge(caseFileDf[["file", "view", "CaseID"]], on="CaseID")
    df_wide.sort_values(by="CaseID", inplace=True)
    filesDf.sort_values(by="CaseID", inplace=True)
    # print(completeDf[["file", "view", "CaseID"]])
    return df_wide, filesDf


def get_label_df(filename):
    labelDf = wr.s3.read_csv(path=f"s3://imgs_labels/{filename}")

    return labelDf


def find_issues_and_correct(expId, vehicleType, view, downloadDir):
    get_cv_pred(expId, vehicleType, downloadDir, view)
    targetCsvName = f"cv_pred_{vehicleType}_{view}.csv"
    completePredDf = combine_df(downloadDir, targetCsvName)
    allParts = completePredDf["part"].unique().tolist()
    # allParts = [x for x in allParts if x not in ["vision_misc", "vision_non_external"]]
    viewPredDf, filesDf = compile_pred_proba(completePredDf, allParts, "conf")
    viewGtDf, filesDf = compile_pred_proba(completePredDf, allParts, "gt")
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

    multilabelDf = wr.s3.read_parquet(
        path=f"s3://multilabel_df/",
        dataset=True,
    )

    targetImgLabelFilename = f"{vehicleType}_{view}_img_labels.csv"
    currentVerImgLabelDf = get_label_df(targetImgLabelFilename)
    topHalfLabelIssue = ranked_label_issues[: len(ranked_label_issues) // 2]
    currentAvailablePartToCorrect = sorted(allParts)
    allCaseWithLabelIssue = viewGtDf.index.values[topHalfLabelIssue]
    allCaseWithLabelIssue
    correctedLabelDf = currentVerImgLabelDf.copy(deep=True)
    for targetPart in currentAvailablePartToCorrect:
        fpCaseToConvertDf = correctedLabelDf[
            (correctedLabelDf["CaseID"].isin(allCaseWithLabelIssue))
            & (correctedLabelDf[targetPart] == 1)
        ][[targetPart, "CaseID"]]
        posWithoutIssue = len(
            correctedLabelDf[
                (~correctedLabelDf["CaseID"].isin(fpCaseToConvertDf["CaseID"]))
                & ((correctedLabelDf[targetPart] == 1))
            ]
        )
        noIssuePosLabelRatio = posWithoutIssue / len(correctedLabelDf)
        if noIssuePosLabelRatio < 0.05:
            logger.warning(f"Skipping {targetPart} due low pos label after filtering")
            continue
        logger.success(
            f"{targetPart} : {noIssuePosLabelRatio} pos label left with no issues"
        )
        """
        Convert labels to zero
        """
        correctedLabelDf.loc[
            correctedLabelDf["CaseID"].isin(
                fpCaseToConvertDf["CaseID"].unique().tolist()
            ),
            targetPart,
        ] = 0

    logger.success(correctedLabelDf)
    logger.success(f"Done correcting {vehicleType} : {view} dataset")
    persist_corrected_dataframe(correctedLabelDf)
    logger.success(f"Done saving {vehicleType} : {view} dataset")


def persist_corrected_dataframe(df):
    imgLabelFilename = f"{vehicleType}_{view}_img_labels.csv"
    wr.s3.to_csv(
        df=df,
        path=f"s3://imgs_labels_corrected/{imgLabelFilename}",
    )


if __name__ == "__main__":
    downloadDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/cleanlab"
    expId = 109

    allVehicleType = ["Saloon - 4 Dr", "Hatchback - 5 Dr", "SUV - 5 Dr"]
    for vehicleType in tqdm(allVehicleType):
        for view in tqdm(get_view_names()):
            find_issues_and_correct(expId, vehicleType, view, downloadDir)
