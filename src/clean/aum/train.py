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
from src.clean.aum.get_files import get_files
from src.clean.aum.get_labels import get_labels
from src.clean.aum.clean import clean_dataset_and_eval
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
import glob


class ProcessModel(pl.LightningModule):
    def __init__(self, model, part, lr, isFiltered=False):
        super(
            ProcessModel,
            self,
        ).__init__()
        pl.seed_everything(99)

        self.model = model
        self.testAccMetric = MulticlassAccuracy(num_classes=2)
        self.trainAccMetric = MulticlassAccuracy(num_classes=2)
        if isFiltered:
            self.aum_dir = f"data/build_dataset/aum/{part}_clean"

        else:
            self.aum_dir = f"data/build_dataset/aum/{part}"

        os.makedirs(self.aum_dir, exist_ok=True)
        self.aum_calculator = AUMCalculator(self.aum_dir, compressed=True)

        self.testConfMat = MulticlassConfusionMatrix(
            num_classes=2, normalize="true"
        ).to(self.device)
        self.testF1 = F1Score(task="multiclass", num_classes=2).to(self.device)
        self.trainConfMat = MulticlassConfusionMatrix(
            num_classes=2, normalize="true"
        ).to(self.device)

        self.testPrecision = Precision(
            task="multiclass",
            num_classes=2,
        ).to(self.device)
        self.testRecall = Recall(task="multiclass", num_classes=2).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pr_curve = MulticlassPrecisionRecallCurve(num_classes=2, thresholds=11)
        self.learning_rate = lr
        self.part = part
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.learning_rate < 1e-2:
            return torch.optim.Adam(self.parameters(), self.learning_rate)
        else:
            return torch.optim.SGD(self.parameters(), self.learning_rate)

    def forward(self, imgs):
        logit = self.model(imgs)
        return logit

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        targets = batch["target"]
        filename = batch["filename"]
        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, dim=1)
        self.aum_calculator.update(logit, targets, filename)

        self.trainAccMetric.update(preds, targets)
        self.trainConfMat.update(preds, targets)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        testAcc = self.trainAccMetric.compute()
        self.log("t_acc", testAcc, prog_bar=True)
        self.trainAccMetric.reset()
        confMat = self.trainConfMat.compute()
        # tn = confMat[0][0]
        # fp = confMat[0][1]

        tp = confMat[1][1]
        tn = confMat[0][0]

        self.log("t_tp", tp, prog_bar=False)
        self.log("t_tn", tn, prog_bar=False)

        # self.log("train_fp", fp, prog_bar=False)
        # self.log("train_fn", fn, prog_bar=False)

        self.trainConfMat.reset()

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, idx):
        imgs = batch["img"]
        targets = batch["target"]
        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, dim=1)
        # subsetAcc = accuracy_score()
        self.testAccMetric.update(preds, targets)
        self.testConfMat.update(preds, targets)
        self.testPrecision.update(preds, targets)
        self.testRecall.update(preds, targets)
        self.testF1.update(preds, targets)

        self.log("e_loss", loss, prog_bar=False)

        return preds, targets

    def validation_epoch_end(self, val_step_outputs) -> None:
        testAcc = self.testAccMetric.compute()
        testPrecision = self.testPrecision.compute()
        testRecall = self.testRecall.compute()
        testF1 = self.testF1.compute()
        self.log("e_acc", testAcc, prog_bar=True)
        self.log("precision", testPrecision, prog_bar=False)
        self.log("recall", testRecall, prog_bar=False)
        self.log("f1", testF1, prog_bar=True)

        confMat = self.testConfMat.compute()

        tp = confMat[1][1]
        tn = confMat[0][0]

        self.log("e_tp", tp, prog_bar=True)
        self.log("e_tn", tn, prog_bar=True)

        self.testConfMat.reset()
        self.testRecall.reset()
        self.testPrecision.reset()
        self.testAccMetric.reset()
        self.testF1.reset()

        return super().validation_epoch_end(val_step_outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        imgs = batch["img"]
        files = batch["filename"]
        logit = self(imgs)
        preds = torch.argmax(logit, dim=1)
        labels = batch["target"].type(torch.uint8)
        predProbs = torch.softmax(logit, dim=1)
        # predProbNp = predProb.cpu().numpy().tolist()[0]
        allPredInfo = []
        for p, f, gt, probs in zip(preds, files, labels, predProbs):
            softmaxProbs = probs.cpu().numpy().tolist()
            info = {
                "pred": p.tolist(),
                "gt": gt.tolist(),
                "probs_0": softmaxProbs[0],
                "probs_1": softmaxProbs[1],
                "filename": f,
                "dataset_index": dataloader_idx,
            }
            allPredInfo.append(info)
        predDf = pd.json_normalize(allPredInfo)
        return predDf


def create_model():

    # load Faster RCNN pre-trained model

    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=2)

    return model


def train_eval(
    trainLoader,
    valLoader,
    testLoader,
    partName,
    lr,
    isFiltered=False,
    train_epoch: int = 10,
):

    checkpoint_callback = ModelCheckpoint(
        monitor="e_acc",
        save_top_k=1,
        mode="max",
        filename="{e_acc:.2f}-{e_tp:.2f}--{e_tn:.2f}",
    )
    model = create_model()

    trainProcessModel = ProcessModel(model, partName, lr, isFiltered)
    trainer1 = pl.Trainer(
        default_root_dir=f"/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/aum_models/",
        max_epochs=train_epoch,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=0,
        benchmark=True,
        precision=16,
        # logger=logger,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
        detect_anomaly=False,
        # limit_train_batches=1,
        # limit_val_batches=
    )

    trainer1.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    trainProcessModel.aum_calculator.finalize()
    predOutputs = trainer1.predict(
        trainProcessModel, dataloaders=[trainLoader, valLoader, testLoader]
    )
    predDf: pd.DataFrame = pd.concat(itertools.chain(*predOutputs))
    if isFiltered:
        outputPredFile = f"{trainProcessModel.aum_dir}/pred_cleaned_{partName}.csv"

    else:
        outputPredFile = f"{trainProcessModel.aum_dir}/pred_ori_{partName}.csv"
    predDf.to_csv(outputPredFile)
    return outputPredFile


def get_dataloader(y_train, y_eval, y_test, targetName, img_dir):
    batchSize = 100
    trainCPUWorker = 10
    imgSize = 300
    trainTransform = A.Compose(
        [
            A.LongestMaxSize(imgSize),
            A.PadIfNeeded(
                min_height=imgSize,
                min_width=imgSize,
                border_mode=0,
            ),
            A.ColorJitter(p=0.2),
            A.Rotate(border_mode=0, p=0.2),
            A.GaussianBlur(blur_limit=(1, 5), p=0.2),
            # A.Downscale(scale_min=0.7, scale_max=0.8, p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    evalTransform = A.Compose(
        [
            A.LongestMaxSize(imgSize),
            A.PadIfNeeded(
                min_height=imgSize,
                min_width=imgSize,
                border_mode=0,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    trainDs = ImageDataset(y_train, img_dir, targetName, trainTransform)

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=batchSize,
        num_workers=trainCPUWorker,
        pin_memory=False,
        persistent_workers=False,
    )
    evalDs = ImageDataset(y_eval, img_dir, targetName, evalTransform)
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=batchSize,
        num_workers=trainCPUWorker,
    )
    testDs = ImageDataset(y_test, img_dir, targetName, evalTransform)
    testLoader = DataLoader2(
        testDs,
        shuffle=False,
        batch_size=batchSize,
        num_workers=trainCPUWorker,
    )
    assert set(evalDs.df["filename"].unique().tolist()).isdisjoint(
        set(trainDs.df["filename"].unique().tolist())
    )
    assert set(testDs.df["filename"].unique().tolist()).isdisjoint(
        set(trainDs.df["filename"].unique().tolist())
    )
    assert set(testDs.df["filename"].unique().tolist()).isdisjoint(
        set(evalDs.df["filename"].unique().tolist())
    )
    return trainLoader, evalLoader, testLoader


def gen_test_label_df(df: pd.DataFrame, targetCols: str):
    testDf = df.groupby(targetCols).sample(500)
    df = df[~df["CaseID"].isin(testDf["CaseID"].unique().tolist())]
    return df, testDf


def fit_clean(labelFile: str, partName: str, imgDir: str, isFiltered=False):
    df = pd.read_csv(labelFile)
    viewPartId = labelFile.split("/")[-1].split(".")[0]
    targetCols = df.filter(regex="vision_*").columns[0]
    df, testDf = gen_test_label_df(df, partName)

    X_train, X_eval, y_train, y_eval = train_test_split(
        df[["filename", "CaseID"]],
        df[targetCols],
        stratify=df[targetCols],
        test_size=0.2,
    )
    y_train = y_train.to_frame()
    y_eval = y_eval.to_frame()
    print(y_eval[[targetCols]].value_counts().reset_index())
    print(y_train[[targetCols]].value_counts().reset_index())

    y_train["filename"] = X_train["filename"]
    y_train["CaseID"] = X_train["CaseID"]

    y_eval["filename"] = X_eval["filename"]
    y_eval["CaseID"] = X_eval["CaseID"]
    y_test = testDf[[targetCols, "filename", "CaseID"]]
    trainLoader, valLoader, testLoader = get_dataloader(
        y_train, y_eval, y_test, partName, imgDir
    )

    origPredFile = train_eval(
        trainLoader,
        valLoader,
        testLoader,
        viewPartId,
        5e-2,
        isFiltered=False,
        train_epoch=10,
    )
    aumCsv = "/".join(origPredFile.split("/")[:-1]) + "/aum_values.csv"
    selectedSamplesCsv = clean_dataset_and_eval(
        origPredFile, aumCsv, selectSamples=True
    )
    selectedSamplesDf = pd.read_csv(selectedSamplesCsv)
    beforeRemove = len(df)
    df = df[df["filename"].isin(selectedSamplesDf["filename"].unique().tolist())]
    afterRemove = len(df)
    diffRatio = (beforeRemove - afterRemove) / beforeRemove
    print(f"Removed : {(beforeRemove - afterRemove)} samples : {diffRatio}")
    X_train, X_eval, y_train, y_eval = train_test_split(
        df[["filename", "CaseID"]],
        df[targetCols],
        stratify=df[targetCols],
        test_size=0.2,
    )
    y_train = y_train.to_frame()
    y_eval = y_eval.to_frame()
    y_train["filename"] = X_train["filename"]
    y_train["CaseID"] = X_train["CaseID"]

    y_eval["filename"] = X_eval["filename"]
    y_eval["CaseID"] = X_eval["CaseID"]
    trainLoader, valLoader, testLoader = get_dataloader(
        y_train, y_eval, y_test, partName, imgDir
    )
    cleanPredFile = train_eval(
        trainLoader,
        valLoader,
        testLoader,
        viewPartId,
        1e-3,
        isFiltered=True,
        train_epoch=10,
    )
    cleanAumCsv = "/".join(cleanPredFile.split("/")[:-1]) + "/aum_values.csv"

    clean_dataset_and_eval(cleanPredFile, cleanAumCsv, selectSamples=False)

    return selectedSamplesDf


def train_original(partName: str):
    partName2 = partName.replace("vision_", "")
    template = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/"
    allViewFilesByPart = glob.glob(f"{template}/{partName2}*.csv", recursive=False)
    imgDir = (
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"
    )
    allLocalFiles = [x.split("/")[-1] for x in glob.glob(f"{imgDir}/**.JPG")]

    for labelFile in allViewFilesByPart:
        # labelFile = f"/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/{partName2}_Rear View_img_labels.csv"
        df = pd.read_csv(labelFile)
        requiredFiles = df["filename"].tolist()
        localReqFiles = set(allLocalFiles) & set(requiredFiles)
        if len(localReqFiles) < len(requiredFiles):
            get_files(labelFile)
        fit_clean(labelFile, partName, imgDir, isFiltered=False)


if __name__ == "__main__":
    allParts = [
        "vision_bonnet",
        "vision_bumper_front",
        "vision_engine",
        "vision_grille",
        "vision_headlamp_lh",
        "vision_headlamp_rh",
        "vision_bumper_rear",
        "vision_front_panel",
        "vision_fender_front_lh",
        "vision_fender_front_rh",
        "vision_rear_quarter_lh",
        "vision_tail_lamp_lh",
        "vision_tail_lamp_rh",
        "vision_windscreen_front",
        "vision_rear_compartment",
        "vision_rear_panel",
        "vision_rear_quarter_rh",
        "vision_windscreen_rear",
    ]
    for p in tqdm(allParts):
        # get_labels(p)

        train_original(p)
