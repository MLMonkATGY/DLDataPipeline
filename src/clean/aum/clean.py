from collections import Counter
from dataclasses import dataclass
from pickle import TRUE
from pprint import pprint
from typing import Any, List, Union
from matplotlib import pyplot
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import albumentations as A

from torch.utils.data import DataLoader2
from torchvision.transforms import transforms
from loguru import logger
from torchvision.datasets import ImageFolder
from torch.cuda.amp.autocast_mode import autocast

from tqdm import tqdm
import numpy as np
import copy
from pytorch_lightning.loggers import MLFlowLogger
from src.analysis.ensemble_predictions import (
    ensemble_pred,
    eval_by_parts,
    get_raw_multilabel_df,
)
from pytorch_lightning.callbacks import ModelCheckpoint


import torchvision

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import MulticlassPrecisionRecallCurve

from torchmetrics.classification.confusion_matrix import (
    ConfusionMatrix,
    MulticlassConfusionMatrix,
)
import pandas as pd
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.metrics import accuracy_score

import awswrangler as wr
from sklearn.model_selection import train_test_split
from src.clean.aum.dataset import ImageDataset
from aum import AUMCalculator
import os
import cleanlab
import itertools
from cleanlab.outlier import OutOfDistribution


def generate_ood_score(predDf: pd.DataFrame):
    ood = OutOfDistribution()
    train_ood_predictions_scores = ood.fit_score(
        pred_probs=predDf[["probs_0", "probs_1"]].values, labels=predDf["gt"].values
    )
    predDf["ood_score"] = train_ood_predictions_scores
    return predDf


def generate_health_summary(predDf: pd.DataFrame, ds_index: int):
    print(f"Dataset index {ds_index} health summary")
    unseenDf = predDf[predDf["dataset_index"] == ds_index]
    healthSummaryDf = cleanlab.dataset.rank_classes_by_label_quality(
        labels=unseenDf["gt"].values, pred_probs=unseenDf[["probs_0", "probs_1"]].values
    )
    print(healthSummaryDf)
    return healthSummaryDf


def merge_aum(predDf: pd.DataFrame, aumDf: pd.DataFrame):
    predDf = predDf.merge(aumDf, left_on="filename", right_on="sample_id", how="left")
    print(predDf)
    return predDf


def select_quality_samples(predDf: pd.DataFrame):
    aumThreshold = predDf["aum"].quantile(0.3)
    if aumThreshold < 0.4:
        aumThreshold = 0.4
    oodScoreThreshold = predDf["ood_score"].quantile(0.1)
    print(f"OOD threshold : {oodScoreThreshold}")
    trainDf = predDf.dropna(subset="aum")
    selectedDf = trainDf[
        (trainDf["aum"] >= aumThreshold) & (trainDf["ood_score"] >= oodScoreThreshold)
    ]
    selectedRatio = len(selectedDf) / len(trainDf)
    print(f"Selected Data :{selectedRatio}")
    print(selectedDf["gt"].value_counts().reset_index())
    print(selectedDf["pred"].value_counts().reset_index())

    return selectedDf


def clean_dataset_and_eval(
    inputPredCsv: str, inputAumCsv: str, selectSamples: bool = False
):
    predDfCsv = inputPredCsv
    aumCsv = inputAumCsv

    outputDir = "/".join(predDfCsv.split("/")[:-1])
    df = pd.read_csv(predDfCsv)
    aumDf = pd.read_csv(aumCsv)
    df = generate_ood_score(df)
    df = merge_aum(df, aumDf)
    testDsHealthSummaryDf = generate_health_summary(df, 2)

    if selectSamples:
        df.to_csv(f"{outputDir}/all_info_before_select.csv")
        selectedDf = select_quality_samples(df)
        selectedSamplesPath = f"{outputDir}/selected.csv"
        testDsHealthSummaryDf.to_csv(f"{outputDir}/test_set_health_summary.csv")
        selectedDf.to_csv(selectedSamplesPath)
        return selectedSamplesPath
    else:
        evalSetDs = generate_health_summary(df, 1)
        evalSetDs.to_csv(f"{outputDir}/eval_set_health_summary.csv")
        df.to_csv(f"{outputDir}/all_info_after_select.csv")

    # df = generate_health_summary(df)
