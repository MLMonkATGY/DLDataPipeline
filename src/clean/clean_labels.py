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
        # "front_view",
        "front_view_right",
        # "rear_view",
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


def count_pos_pred(row, targetParts: List[str]):
    targetPreds = np.array(row[targetParts].values > 0.5)
    posCount = targetPreds.sum()
    return posCount


def count_pos_gt(row, targetParts: List[str]):
    posCount = int(row[targetParts].values.sum())
    return posCount


def get_top_issue(expId, vehicleType, view, downloadDir, topN):
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
    issueCoverage = len(ranked_label_issues) / len(viewGtDf)
    logger.info(f"Issue covers : {issueCoverage} of total case")
    topIssues = ranked_label_issues[:topN]
    allCaseWithLabelIssue = viewGtDf.index.values[topIssues]
    return allCaseWithLabelIssue


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
    topHalfLabelIssue = ranked_label_issues
    targetPartToCorrect = [
        # "vision_bonnet",
        # "vision_grille",
        "vision_bumper_front",
        "vision_engine",
        "vision_bumper_rear",
        "vision_wheel",
        "vision_headlamp_lh",
        "vision_headlamp_rh",
        # "vision_fender_front_lh",
        # "vision_fender_front_rh",
        # "vision_rear_quarter_lh",
        # "vision_rear_quarter_rh",
        "vision_rear_compartment",
    ]
    currentAvailablePartToCorrect = sorted(
        list(set(targetPartToCorrect) & set(allParts))
    )

    # currentAvailablePartToCorrect = sorted(allParts)
    allCaseWithLabelIssue = viewGtDf.index.values[topHalfLabelIssue]

    correctedLabelDf = currentVerImgLabelDf.copy(deep=True)
    viewPredDf["pred_pos_num"] = viewPredDf.apply(
        lambda x: count_pos_pred(x, currentAvailablePartToCorrect), axis=1
    )
    exclude = ["vision_misc", "vision_non_external"]
    targetsGtParts = list(
        set(viewGtDf.filter(regex="vision_*").columns).difference(set(exclude))
    )

    viewGtDf["gt_pos_num"] = viewGtDf.apply(
        lambda x: count_pos_gt(x, targetsGtParts), axis=1
    )
    tempDf = viewGtDf[["gt_pos_num"]].merge(
        viewPredDf[["pred_pos_num"]], left_index=True, right_index=True
    )
    # limit1 = len(tempDf[tempDf["pred_pos_num"] <= 1])
    # limit2 = len(tempDf[tempDf["gt_pos_num"] >= 3])
    # print(len(tempDf[tempDf["pred_pos_num"] <= 1]))
    # print(len(tempDf[tempDf["gt_pos_num"] >= 3]))
    # print(tempDf["pred_pos_num"].value_counts(normalize=True))
    # print(tempDf["gt_pos_num"].value_counts(normalize=True))

    susFpCaseId = tempDf[
        (tempDf["gt_pos_num"] >= 3) & (tempDf["pred_pos_num"] <= 1)
    ].index.values
    # print(susFpCaseId)
    fpCaseId = np.intersect1d(susFpCaseId, allCaseWithLabelIssue).tolist()
    susFnCaseId = tempDf[
        (tempDf["gt_pos_num"] <= 1) & (tempDf["pred_pos_num"] >= 3)
    ].index.values
    fnCaseId = np.intersect1d(susFnCaseId, allCaseWithLabelIssue).tolist()

    for targetPart in currentAvailablePartToCorrect:
        oriPosSampleCount = len(correctedLabelDf[correctedLabelDf[targetPart] == 1])
        oriNegSampleCount = len(correctedLabelDf[correctedLabelDf[targetPart] == 0])

        fpCaseToConvertDf = correctedLabelDf[
            (correctedLabelDf["CaseID"].isin(fpCaseId))
            # & (correctedLabelDf[targetPart] == 1)
        ][[targetPart, "CaseID"]]

        # posWithoutIssue = len(
        #     correctedLabelDf[
        #         (~correctedLabelDf["CaseID"].isin(susFpCaseId))
        #         & ((correctedLabelDf[targetPart] == 1))
        #     ]
        # )
        # noIssuePosLabelRatio = posWithoutIssue / len(correctedLabelDf)
        # if noIssuePosLabelRatio < 0.05:
        #     logger.warning(f"Skipping {targetPart} due low pos label after filtering")
        #     continue
        # logger.success(
        #     f"{targetPart} : {noIssuePosLabelRatio} pos label left with no issues"
        # )
        """
        Convert labels to zero
        """
        correctedLabelDf.loc[
            correctedLabelDf["CaseID"].isin(
                fpCaseToConvertDf["CaseID"].unique().tolist()
            ),
            targetPart,
        ] = 0
        # fnCaseToConvertDf = correctedLabelDf[
        #     (correctedLabelDf["CaseID"].isin(fnCaseId))
        #     # & (correctedLabelDf[targetPart] == 0)
        # ][[targetPart, "CaseID"]]
        # correctedLabelDf.loc[
        #     correctedLabelDf["CaseID"].isin(
        #         fnCaseToConvertDf["CaseID"].unique().tolist()
        #     ),
        #     targetPart,
        # ] = 1
        afterCorrectionPosCount = len(
            correctedLabelDf[correctedLabelDf[targetPart] == 1]
        )
        afterCorrectionNegCount = len(
            correctedLabelDf[correctedLabelDf[targetPart] == 0]
        )
    # t = correctedLabelDf[
    #     (correctedLabelDf["CaseID"].isin(fpCaseId))
    #     # & (correctedLabelDf[targetPart] == 1.0)
    # ]
    # print(correctedLabelDf["CaseID"].describe())
    # print(correctedLabelDf[targetPart])

    # print(t)
    # logger.success(correctedLabelDf)
    # fpCorrected = len(
    #     correctedLabelDf[
    #         (correctedLabelDf[targetPart] == 1)
    #         & (correctedLabelDf["CaseID"].isin(fpCaseId))
    #         # & (correctedLabelDf[targetPart] == 1.0)
    #     ]
    # )
    # fnCorrected = len(
    #     correctedLabelDf[
    #         (correctedLabelDf["CaseID"].isin(fnCaseId))
    #         & (correctedLabelDf[targetPart] == 0)
    #     ]
    # )
    logger.success(f"Detected FP error {afterCorrectionPosCount - oriPosSampleCount}")
    logger.success(f"Detected FN error {oriNegSampleCount - afterCorrectionNegCount}")

    logger.success(f"Done correcting {vehicleType} : {view} dataset")
    persist_corrected_dataframe(correctedLabelDf)
    logger.success(f"Done saving {vehicleType} : {view} dataset")


def persist_corrected_dataframe(df):
    imgLabelFilename = f"{vehicleType}_{view}_img_labels.csv"
    wr.s3.to_csv(
        df=df,
        path=f"s3://imgs_labels_corrected_2/{imgLabelFilename}",
    )


if __name__ == "__main__":
    downloadDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/cleanlab/v2"
    expId = 113

    allVehicleType = ["Saloon - 4 Dr", "Hatchback - 5 Dr", "SUV - 5 Dr"]
    allCaseWithIssue = []
    for vehicleType in tqdm(allVehicleType):
        for view in tqdm(get_view_names()):
            topIssueCaseId = get_top_issue(
                expId, vehicleType, view, downloadDir, topN=1500
            )
            allCaseWithIssue.extend(topIssueCaseId)
    caseStudyDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/case_study/case_study_5/case_study_result.csv"
    )
    caseStudyCaseId = set(caseStudyDf["CaseID"].unique().tolist())
    uniqueIssueCaseToRemove = set(allCaseWithIssue).difference(caseStudyCaseId)
    logger.success(f"Removing {len(uniqueIssueCaseToRemove) / 30e3} of cases")
    noisyLabelDf = pd.DataFrame({"CaseID": sorted(list(uniqueIssueCaseToRemove))})
    noisyLabelDf.to_csv("./noisy_label_top15.csv")
