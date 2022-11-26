from dataclasses import dataclass
from pickle import TRUE
from pprint import pprint
from typing import Any, List
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

import os
import ujson as json
from tqdm import tqdm
import numpy as np
import copy
from pytorch_lightning.loggers import MLFlowLogger
from ClassifierDataset import DmgClassifierDataset
from TrainClassifierParams import trainParams
from pytorch_lightning.callbacks import ModelCheckpoint

from data import ImportEnv
from loguru import logger as displayLogger
from mlflow.tracking.client import MlflowClient
import itertools
import warnings
import torchvision
import torchmetrics
import dataclasses
import shutil
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.calibration_error import MulticlassCalibrationError

from torchmetrics import Precision, Recall
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import glob
from cleanlab.classification import CleanLearning
import pathlib
from Pipeline import getAllParts
import mlflow


def create_model():

    # load Faster RCNN pre-trained model
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=2)

    return model


def GetDataloaders(trainFile, valFile):
    trainTransform = A.Compose(
        [
            A.LongestMaxSize(trainParams.imgSize),
            A.PadIfNeeded(
                min_height=trainParams.imgSize,
                min_width=trainParams.imgSize,
                border_mode=0,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    evalTransform = A.Compose(
        [
            A.LongestMaxSize(trainParams.imgSize),
            A.PadIfNeeded(
                min_height=trainParams.imgSize,
                min_width=trainParams.imgSize,
                border_mode=0,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    trainDs = DmgClassifierDataset(
        trainParams.srcAnnFile, trainParams.targetPart, trainTransform, trainFile
    )

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=trainParams.trainBatchSize,
        num_workers=trainParams.trainCPUWorker,
    )
    evalDs = DmgClassifierDataset(
        trainParams.srcAnnFile, trainParams.targetPart, evalTransform, valFile
    )
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=trainParams.trainBatchSize,
        num_workers=5,
    )
    testDs = DmgClassifierDataset(
        trainParams.srcAnnFile, trainParams.targetPart, evalTransform, valFile
    )
    testLoader = DataLoader2(
        testDs,
        shuffle=False,
        batch_size=1,
        num_workers=5,
    )
    assert set(evalDs.df["Filename"].unique().tolist()).isdisjoint(
        set(trainDs.df["Filename"].unique().tolist())
    )
    assert set(testDs.df["Filename"].unique().tolist()).isdisjoint(
        set(trainDs.df["Filename"].unique().tolist())
    )
    return trainLoader, evalLoader, testLoader


class ProcessModel(pl.LightningModule):
    def __init__(self):
        super(ProcessModel, self).__init__()
        self.model = create_model()
        self.testAccMetric = Accuracy(num_classes=2)
        self.trainAccMetric = Accuracy(num_classes=2)
        self.testConfMat = ConfusionMatrix(num_classes=2, normalize="true").to(
            self.device
        )
        self.trainConfMat = ConfusionMatrix(num_classes=2, normalize="true").to(
            self.device
        )
        self.testCalibrationError = MulticlassCalibrationError(num_classes=2).to(
            self.device
        )
        self.testPrecision = Precision(num_classes=2, average="macro").to(self.device)
        self.testRecall = Recall(num_classes=2, average="macro").to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), trainParams.learningRate)

    def forward(self, imgs):
        logit = self.model(imgs)
        return logit

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        targets = batch["target"]

        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, 1)
        self.trainAccMetric.update(preds, targets)
        self.trainConfMat.update(preds, targets)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        testAcc = self.trainAccMetric.compute()
        self.log("train_acc", testAcc, prog_bar=True)
        self.trainAccMetric.reset()
        confMat = self.trainConfMat.compute()
        tn = confMat[0][0]
        fp = confMat[0][1]
        tp = confMat[1][1]
        fn = confMat[1][0]
        self.log("train_tp", tp, prog_bar=False)
        self.log("train_tn", tn, prog_bar=False)
        self.log("train_fp", fp, prog_bar=False)
        self.log("train_fn", fn, prog_bar=False)

        self.trainConfMat.reset()

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, idx):
        imgs = batch["img"]
        targets = batch["target"]
        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, 1)
        self.testAccMetric.update(preds, targets)
        self.testConfMat.update(preds, targets)
        self.testPrecision.update(preds, targets)
        self.testRecall.update(preds, targets)
        self.testCalibrationError.update(logit, targets)

    def validation_epoch_end(self, val_step_outputs) -> None:
        testAcc = self.testAccMetric.compute()
        testCalibrationError = self.testCalibrationError.compute()
        testPrecision = self.testPrecision.compute()
        testRecall = self.testRecall.compute()

        self.log("test_acc", testAcc, prog_bar=True)
        self.log("test_calibration_error", testCalibrationError, prog_bar=False)
        self.log("test_precision", testPrecision, prog_bar=False)
        self.log("test_recall", testRecall, prog_bar=False)

        confMat = self.testConfMat.compute()
        tn = confMat[0][0]
        fp = confMat[0][1]
        tp = confMat[1][1]
        fn = confMat[1][0]
        self.log("test_tp", tp, prog_bar=True)
        self.log("test_tn", tn, prog_bar=True)
        self.log("test_fp", fp, prog_bar=False)
        self.log("test_fn", fn, prog_bar=False)

        self.testConfMat.reset()
        self.testRecall.reset()
        self.testPrecision.reset()
        self.testAccMetric.reset()

        return super().validation_epoch_end(val_step_outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch["img"]
        file = batch["file"]
        label = batch["target"].cpu().numpy()
        logit = self(imgs)
        predictedCls = torch.argmax(logit, dim=1).cpu().numpy()
        predProb = F.softmax(logit, dim=1)
        predProbNp = predProb.cpu().numpy().tolist()[0]
        preds = {
            "pred_probs": predProbNp,
            "file": file,
            "gt": label,
            "pred": predictedCls,
        }
        return preds


def trainEval(trainFile, valFile, kfoldId):
    trainLoader, valLoader, testLoader = GetDataloaders(trainFile, valFile)

    logger = MLFlowLogger(
        experiment_name=trainParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=f"{trainParams.imgAngle}_{trainParams.runName}_{kfoldId}",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="test_acc",
        save_top_k=trainParams.saveTopNBest,
        mode="max",
        filename="{epoch:02d}-{test_acc:.2f}",
    )
    trainProcessModel = ProcessModel()
    trainer = pl.Trainer(
        # accumulate_grad_batches=5,
        default_root_dir=f"./outputs/{trainParams.localSaveDir}",
        max_epochs=trainParams.maxEpoch,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=trainParams.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        benchmark=True,
        precision=trainParams.trainingPrecision,
        logger=logger,
        log_every_n_steps=30,
        callbacks=[checkpoint_callback],
        detect_anomaly=False,
        # limit_train_batches=3,
        # limit_val_batches=3,
        # limit_predict_batches=100,
    )

    trainer.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    predictions = trainer.predict(trainProcessModel, testLoader)

    displayLogger.success("Started Uploading Best Checkpoints..")
    mlflowLogger: MlflowClient = trainProcessModel.logger.experiment
    mlflowLogger.log_dict(
        trainProcessModel.logger.run_id,
        dataclasses.asdict(trainParams),
        "hyperparams.json",
    )

    mlflowLogger.log_artifacts(
        trainProcessModel.logger.run_id,
        checkpoint_callback.dirpath.replace("/checkpoints", "/"),
    )
    return predictions


def BalanceSampling(srcDf, targetPart):
    labelDistribDf = srcDf[targetPart].value_counts().reset_index()
    smallestClassSize = labelDistribDf[targetPart].min()
    # print(srcDf.index)
    srcDf2 = srcDf.groupby(targetPart).sample(n=smallestClassSize).reset_index()
    # print(srcDf2.index)

    # print(srcDf2[targetPart].value_counts())
    return srcDf2


def GetDataDir():
    currPath = pathlib.Path(os.getcwd())
    imgSrcDir = currPath / "data/train_test_img"
    labelSrcDir = currPath / "data/train_test_labels"
    assert os.path.isdir(imgSrcDir)
    assert os.path.isdir(labelSrcDir)

    return imgSrcDir, labelSrcDir


def getAllPart():
    currPath = pathlib.Path(os.getcwd())
    allPartPath = currPath / "data/all_part.json"
    assert os.path.exists(allPartPath)
    with open(allPartPath, "r") as f:
        allParts = json.load(f)
    return allParts


def GenerateNewRunForCrossValPred(allPreds: List, targetPart: str):
    run_name = f"cross_val_pred_{targetPart}"
    outputDir = pathlib.Path(os.getcwd()) / "outputs"
    os.makedirs(outputDir, exist_ok=True)
    with mlflow.start_run(experiment_id=trainParams.expId, run_name=run_name):
        crossValPredDf = pd.json_normalize(allPreds)
        outputName = f"{outputDir}/cross_val_pred_{targetPart}.csv"
        crossValPredDf.to_csv(outputName)
        mlflow.log_artifact(outputName)
        print(f"Completed part {targetPart}")


if __name__ == "__main__":
    imgSrcDir, labelSrcDir = GetDataDir()
    allParts = getAllPart()
    inputDir = labelSrcDir
    searchImgView = f"{inputDir}/*.csv"
    allSrcAnnFile = glob.glob(searchImgView, recursive=True)
    for srcAnnPath in tqdm(allSrcAnnFile, desc="view"):
        imgAngle = srcAnnPath.split("/")[-1].split("_")[1]
        trainParams.srcAnnFile = srcAnnPath
        trainParams.imgAngle = imgAngle
        srcDf = pd.read_csv(trainParams.srcAnnFile)
        allTargetParts = [x for x in srcDf.columns if x in allParts]
        for part in tqdm(allTargetParts, desc="part"):
            trainParams.runName = part
            trainParams.targetPart = part
            skf = StratifiedKFold(n_splits=trainParams.kFoldSplit)
            targetPart = trainParams.targetPart
            srcDf2 = BalanceSampling(srcDf, targetPart)
            y = srcDf2[targetPart]
            X = srcDf2["Path"]
            allSplit = []
            allPreds = []
            for kfoldId, (train_index, test_index) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                labelCount = y_test.value_counts().reset_index()
                print(labelCount)
                trainParams.dmg_label_count = labelCount[labelCount["index"] == 1][
                    targetPart
                ].item()
                trainParams.not_dmg_label_count = labelCount[labelCount["index"] == 0][
                    targetPart
                ].item()

                allSplit.append(
                    {
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                    }
                )

                predictions = trainEval(X_train, X_test, kfoldId + 1)
                allPreds.extend(predictions)
            GenerateNewRunForCrossValPred(allPreds, part)
