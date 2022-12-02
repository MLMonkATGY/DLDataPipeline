from collections import Counter
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
from ClassifierDataset import MultilabelDataset
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
from torchmetrics.classification.accuracy import MultilabelAccuracy
from torchmetrics import Precision, Recall

from torchmetrics.classification.confusion_matrix import (
    ConfusionMatrix,
    MultilabelConfusionMatrix,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import glob
from cleanlab.classification import CleanLearning
import pathlib
from LocalPipeline import getAllParts
import mlflow
from sklearn.metrics import accuracy_score
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import (
    iterative_train_test_split,
)
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class FocalLoss2d(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        gamma=2,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        balance_param=0.25,
    ):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):

        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # compute the negative likelyhood
        logpt = -F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.weight, reduction=self.reduction
        )
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        # balanced_focal_loss = self.balance_param * focal_loss
        return focal_loss


def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        elif len(set_true) == 0:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_pred))
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true))
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list), acc_list


def create_model():

    # load Faster RCNN pre-trained model
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(
        in_features=num_ftrs, out_features=len(trainParams.targetPart)
    )

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
            A.ColorJitter(p=0.2),
            A.CoarseDropout(max_height=16, max_width=16, p=0.2),
            A.GaussianBlur(blur_limit=(1, 5), p=0.2),
            A.Downscale(scale_min=0.6, scale_max=0.8, p=0.2),
            A.GridDistortion(border_mode=0, p=0.2),
            A.RandomGridShuffle(p=0.2),
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
    trainDs = MultilabelDataset(
        trainParams.srcAnnFile, trainParams.targetPart, trainTransform, trainFile
    )

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=trainParams.trainBatchSize,
        num_workers=trainParams.trainCPUWorker,
    )
    evalDs = MultilabelDataset(
        trainParams.srcAnnFile, trainParams.targetPart, evalTransform, valFile
    )
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=trainParams.trainBatchSize,
        num_workers=5,
    )
    testDs = MultilabelDataset(
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
    def __init__(self, pos_weight=None):
        super(ProcessModel, self).__init__()
        self.model = create_model()
        self.testAccMetric = MultilabelAccuracy(num_labels=len(trainParams.targetPart))
        self.trainAccMetric = MultilabelAccuracy(num_labels=len(trainParams.targetPart))
        self.testConfMat = MultilabelConfusionMatrix(
            num_labels=len(trainParams.targetPart), normalize="true"
        ).to(self.device)
        self.trainConfMat = MultilabelConfusionMatrix(
            num_labels=len(trainParams.targetPart), normalize="true"
        ).to(self.device)

        self.testPrecision = Precision(
            num_classes=len(trainParams.targetPart), multiclass=False
        ).to(self.device)
        self.testRecall = Recall(
            num_classes=len(trainParams.targetPart), multiclass=False
        ).to(self.device)

        self.criterion = FocalLoss2d()
        self.sigmoid = torch.nn.Sigmoid()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), trainParams.learningRate)

    def forward(self, imgs):
        logit = self.model(imgs)
        logit = self.sigmoid(logit)
        return logit

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        targets = batch["target"]

        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = logit > 0.5
        self.trainAccMetric.update(preds, targets)
        self.trainConfMat.update(preds, targets.type(torch.int64))
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        testAcc = self.trainAccMetric.compute()
        self.log("train_acc", testAcc, prog_bar=True)
        self.trainAccMetric.reset()
        confMat = self.trainConfMat.compute()
        # tn = confMat[0][0]
        # fp = confMat[0][1]
        # tp = confMat[1][1]
        # fn = confMat[1][0]
        # self.log("train_tp", tp, prog_bar=False)
        # self.log("train_tn", tn, prog_bar=False)
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
        # loss = self.criterion(logit, targets)
        preds = logit > 0.5
        # subsetAcc = accuracy_score()
        self.testAccMetric.update(preds, targets)
        self.testConfMat.update(preds, targets.type(torch.int64))
        self.testPrecision.update(preds, targets.type(torch.int64))
        self.testRecall.update(preds, targets.type(torch.int64))
        return preds, targets

    def validation_epoch_end(self, val_step_outputs) -> None:
        testAcc = self.testAccMetric.compute()
        testPrecision = self.testPrecision.compute()
        testRecall = self.testRecall.compute()
        preds = torch.cat([x[0] for x in val_step_outputs])
        targets = torch.cat([x[1] for x in val_step_outputs])

        exactAcc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        hammingScore, _ = hamming_score(targets.cpu().numpy(), preds.cpu().numpy())
        self.log("test_acc", testAcc, prog_bar=True)
        self.log("precision", testPrecision, prog_bar=True)
        self.log("recall", testRecall, prog_bar=True)
        self.log("exact", exactAcc, prog_bar=True)
        self.log("subset", hammingScore, prog_bar=True)

        confMat = self.testConfMat.compute()
        # tn = confMat[0][0]
        # fp = confMat[0][1]
        # tp = confMat[1][1]
        # fn = confMat[1][0]
        # self.log("test_tp", tp, prog_bar=True)
        # self.log("test_tn", tn, prog_bar=True)
        # self.log("test_fp", fp, prog_bar=False)
        # self.log("test_fn", fn, prog_bar=False)

        self.testConfMat.reset()
        self.testRecall.reset()
        self.testPrecision.reset()
        self.testAccMetric.reset()

        return super().validation_epoch_end(val_step_outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch["img"]
        file = batch["file"]
        rawParts = batch["parts"]
        parts = [x[0] for x in rawParts]
        labels = batch["target"].cpu().numpy().squeeze(0)
        logit = self(imgs)
        preds = (logit > 0.5).cpu().numpy().squeeze(0)
        # predProbNp = predProb.cpu().numpy().tolist()[0]
        info = {
            "file": file,
        }
        for part, pred, label in zip(parts, preds, labels):
            info[f"pred_{part}"] = pred
            info[f"gt_{part}"] = True if label == 1 else False

        return info


def trainEval(trainFile, valFile, kfoldId):
    trainLoader, valLoader, testLoader = GetDataloaders(trainFile, valFile)

    logger = MLFlowLogger(
        experiment_name=trainParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=f"{trainParams.imgAngle}_{kfoldId}",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="test_acc",
        save_top_k=trainParams.saveTopNBest,
        mode="max",
        filename="{epoch:02d}-{test_acc:.2f}",
    )
    trainProcessModel = ProcessModel(trainLoader.dataset.allPosWeight)
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
        log_every_n_steps=100,
        callbacks=[checkpoint_callback],
        detect_anomaly=False,
        # limit_train_batches=1,
        # limit_val_batches=100,
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


def StorePred(allPreds: List, targetPart: List[str], imgAngle: str):
    run_name = f"cv_pred_{imgAngle}"
    outputDir = pathlib.Path(os.getcwd()) / "outputs"
    os.makedirs(outputDir, exist_ok=True)
    with mlflow.start_run(experiment_id=trainParams.expId, run_name=run_name):
        crossValPredDf = pd.json_normalize(allPreds)
        print(crossValPredDf)
        outputName = f"{outputDir}/cv_pred_{imgAngle}.csv"
        crossValPredDf.to_csv(outputName)
        mlflow.log_artifact(outputName)
        print(f"Completed part {imgAngle} {targetPart}")


def GenLabelGroup(x):
    rawNpArray = np.array2string(
        x.values, precision=2, separator=",", suppress_small=True
    )
    return rawNpArray


def CountLabelComb(srcDf, allTargetParts):
    srcDf["label_combination"] = srcDf[allTargetParts].apply(GenLabelGroup, axis=1)
    combDf = srcDf["label_combination"].value_counts().reset_index()
    print(combDf)


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
        trainParams.runName = imgAngle
        trainParams.targetPart = allTargetParts
        mulitlearnStratify = MultilabelStratifiedKFold(n_splits=trainParams.kFoldSplit)
        targetPart = trainParams.targetPart
        # srcDf2 = BalanceSampling(srcDf, allTargetParts)
        y = srcDf[allTargetParts]
        X = srcDf["Path"]
        CountLabelComb(srcDf, allTargetParts)

        allSplit = []
        allPreds = []
        for kfoldId, (train_index, test_index) in enumerate(
            mulitlearnStratify.split(X, y)
        ):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
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
        StorePred(allPreds, targetPart, imgAngle)
