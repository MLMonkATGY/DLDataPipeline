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
from src.analysis.EnsemblePredict import ensemble_pred, eval_by_parts
from dataset import MultilabelDataset
from src.TrainClassifierParams import trainParams
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
import mlflow
from sklearn.metrics import accuracy_score
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import (
    iterative_train_test_split,
)
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
import awswrangler as wr
import optuna

wr.config.s3_endpoint_url = "http://192.168.1.7:8333"


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


def get_dataloader(y_train, y_test):
    trainTransform = A.Compose(
        [
            A.LongestMaxSize(trainParams.imgSize),
            A.PadIfNeeded(
                min_height=trainParams.imgSize,
                min_width=trainParams.imgSize,
                border_mode=0,
            ),
            # A.ColorJitter(p=0.2),
            # A.CoarseDropout(max_height=8, max_width=8, p=0.2),
            # A.GaussianBlur(blur_limit=(1, 5), p=0.2),
            # A.Downscale(scale_min=0.6, scale_max=0.8, p=0.2),
            # A.GridDistortion(border_mode=0, p=0.2),
            # A.RandomGridShuffle(p=0.2),
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
    trainDs = MultilabelDataset(y_train, trainTransform)

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=trainParams.trainBatchSize,
        num_workers=trainParams.trainCPUWorker,
        pin_memory=True,
        persistent_workers=True,
    )
    evalDs = MultilabelDataset(y_test, evalTransform)
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=trainParams.trainBatchSize * 2,
        num_workers=trainParams.trainCPUWorker,
    )
    testDs = MultilabelDataset(y_test, evalTransform)
    testLoader = DataLoader2(
        testDs,
        shuffle=False,
        batch_size=trainParams.trainBatchSize * 2,
        num_workers=trainParams.trainCPUWorker,
    )
    assert set(evalDs.df["filename"].unique().tolist()).isdisjoint(
        set(trainDs.df["filename"].unique().tolist())
    )
    assert set(testDs.df["filename"].unique().tolist()).isdisjoint(
        set(trainDs.df["filename"].unique().tolist())
    )
    return trainLoader, evalLoader, testLoader


class ProcessModel(pl.LightningModule):
    def __init__(self, model, isFineTune=False, pos_weight=None):
        super(
            ProcessModel,
            self,
        ).__init__()
        self.model = model
        self.isFinetune = isFineTune
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

        self.criterion = torch.nn.BCEWithLogitsLoss(
            # pos_weight=torch.tensor(pos_weight) * trainParams.posWeightScaler
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.posThreshold = trainParams.posThreshold
        # self.save_hyperparameters()

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), trainParams.learningRate)

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
        preds = logit > self.posThreshold
        self.trainAccMetric.update(preds, targets)
        self.trainConfMat.update(preds, targets.type(torch.int64))
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        testAcc = self.trainAccMetric.compute()
        self.log("t_acc", testAcc, prog_bar=False)
        self.trainAccMetric.reset()
        confMat = self.trainConfMat.compute()
        # tn = confMat[0][0]
        # fp = confMat[0][1]
        allTp = []
        allTn = []

        for i, clsMat in enumerate(confMat):
            tp = clsMat[1][1]
            tn = clsMat[0][0]
            self.log(f"t_cls_{i}_tp", tp, prog_bar=False)
            self.log(f"t_cls_{i}_tn", tn, prog_bar=False)
            allTp.append(tp)
            allTn.append(tn)
        avgTp = torch.mean(torch.tensor(allTp))
        avgTn = torch.mean(torch.tensor(allTn))

        self.log("t_tp", avgTp, prog_bar=True)
        self.log("t_tn", avgTn, prog_bar=True)

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
        preds = logit > self.posThreshold
        # subsetAcc = accuracy_score()
        self.testAccMetric.update(preds, targets)
        self.testConfMat.update(preds, targets.type(torch.int64))
        self.testPrecision.update(preds, targets.type(torch.int64))
        self.testRecall.update(preds, targets.type(torch.int64))
        self.log("e_loss", loss, prog_bar=True)

        return preds, targets

    def validation_epoch_end(self, val_step_outputs) -> None:
        testAcc = self.testAccMetric.compute()
        testPrecision = self.testPrecision.compute()
        testRecall = self.testRecall.compute()
        preds = torch.cat([x[0] for x in val_step_outputs])
        targets = torch.cat([x[1] for x in val_step_outputs])

        exactAcc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        hammingScore, _ = hamming_score(targets.cpu().numpy(), preds.cpu().numpy())
        self.log("e_acc", testAcc, prog_bar=False)
        self.log("precision", testPrecision, prog_bar=False)
        self.log("recall", testRecall, prog_bar=False)
        self.log("exact", exactAcc, prog_bar=False)
        self.log("subset", hammingScore, prog_bar=True)

        confMat = self.testConfMat.compute()
        allTp = []
        allTn = []

        for i, clsMat in enumerate(confMat):
            tp = clsMat[1][1]
            tn = clsMat[0][0]
            self.log(f"e_cls_{i}_tp", tp, prog_bar=False)
            self.log(f"e_cls_{i}_tn", tn, prog_bar=False)
            allTp.append(tp)
            allTn.append(tn)
        avgTp = torch.mean(torch.tensor(allTp))
        avgTn = torch.mean(torch.tensor(allTn))

        self.log("e_tp", avgTp, prog_bar=True)
        self.log("e_tn", avgTn, prog_bar=True)

        self.testConfMat.reset()
        self.testRecall.reset()
        self.testPrecision.reset()
        self.testAccMetric.reset()

        return super().validation_epoch_end(val_step_outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch["img"]
        files = batch["file"]
        parts = [x[0] for x in batch["parts"]]
        labels = batch["target"].cpu().numpy()
        logit = self(imgs)
        preds = (logit > self.posThreshold).cpu().numpy()
        # predProbNp = predProb.cpu().numpy().tolist()[0]
        allDf = []
        for p, f, conf, gt in zip(preds, files, logit, labels):
            info = {
                "file": f,
                "pred": p.tolist(),
                "conf": conf.cpu().numpy().tolist(),
                "parts": parts,
                "gt": gt.tolist(),
            }
            df = pd.DataFrame(info)
            allDf.append(df)

        predDf = pd.concat(allDf)
        return predDf


def train_eval(trainLoader, valLoader, testLoader, kfoldId, posWeight):
    logger = MLFlowLogger(
        experiment_name=trainParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=f"{trainParams.imgAngle}_{kfoldId}",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="subset",
        save_top_k=trainParams.saveTopNBest,
        mode="max",
        filename="{subset:.2f}-{e_tp:.2f}--{e_tn:.2f}",
    )
    model = create_model()
    trainProcessModel = ProcessModel(model, isFineTune=False, pos_weight=posWeight)
    trainer1 = pl.Trainer(
        # accumulate_grad_batches=10,
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
        # limit_train_batches=200,
        # limit_val_batches=5,
        # limit_predict_batches=100,
    )

    trainer1.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    batchPredDf = trainer1.predict(trainProcessModel, testLoader)
    completePredDf = pd.concat(batchPredDf)

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
    return completePredDf


def BalanceSampling(srcDf, targetPart):
    labelDistribDf = srcDf[targetPart].value_counts().reset_index()
    smallestClassSize = labelDistribDf[targetPart].min()
    srcDf2 = srcDf.groupby(targetPart).sample(n=smallestClassSize).reset_index()

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


def store_pred(completePred: pd.DataFrame):
    run_name = f"cv_pred_{trainParams.imgAngle}"
    outputDir = pathlib.Path(os.getcwd()) / "outputs"
    os.makedirs(outputDir, exist_ok=True)
    with mlflow.start_run(experiment_id=trainParams.expId, run_name=run_name):
        outputName = f"{outputDir}/cv_pred_{trainParams.imgAngle}.csv"
        completePred.to_csv(outputName)
        mlflow.log_artifact(outputName)
        print(f"Completed part {trainParams.imgAngle}")


def GenLabelGroup(x):
    rawNpArray = np.array2string(
        x.values, precision=2, separator=",", suppress_small=True
    )
    return rawNpArray


def CountLabelComb(srcDf, allTargetParts):
    srcDf["label_combination"] = srcDf[allTargetParts].apply(GenLabelGroup, axis=1)
    combDf = srcDf["label_combination"].value_counts().reset_index()
    print(combDf)


def get_view_filename():
    return [
        "front_view_left_img_labels.csv",
        "front_view_img_labels.csv",
        "front_view_right_img_labels.csv",
        "rear_view_img_labels.csv",
        "rear_view_left_img_labels.csv",
        "rear_view_right_img_labels.csv",
    ]


def get_label_df(filename):
    labelDf = wr.s3.read_csv(path=f"s3://imgs_labels/{filename}")
    return labelDf


def gen_dataset(viewFilename, notLabels):
    labelDf: pd.DataFrame = get_label_df(viewFilename)
    allTargetParts = [x for x in labelDf.columns if x not in notLabels]
    trainParams.runName = viewFilename
    trainParams.targetPart = allTargetParts
    trainParams.imgAngle = viewFilename.replace("_img_labels.csv", "")
    mulitlearnStratify = MultilabelStratifiedKFold(n_splits=2, shuffle=True)
    y = labelDf[allTargetParts]
    X = labelDf["filename"]
    return mulitlearnStratify, X, y, allTargetParts


def fit(trial, viewFilename):
    notLabels = ["CaseID", "view", "filename", "Unnamed: 0"]

    mulitlearnStratify, X, y, allTargetParts = gen_dataset(viewFilename, notLabels)
    allColName = y.columns.tolist()
    posWeight = [
        trial.suggest_float(f"pos_weight_{allColName[x]}", 0.05, 20, log=True)
        for x in range(len(allColName))
    ]
    viewCompletePredDf = pd.DataFrame()
    for kfoldId, (train_index, test_index) in enumerate(mulitlearnStratify.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        y_train["filename"] = X_train
        y_test["filename"] = X_test
        trainLoader, valLoader, testLoader = get_dataloader(y_train, y_test)
        predDf = train_eval(trainLoader, valLoader, testLoader, posWeight, kfoldId + 1)
        viewCompletePredDf = pd.concat([viewCompletePredDf, predDf])
    store_pred(viewCompletePredDf)
    accDf, partPredDf, allParts = ensemble_pred(viewCompletePredDf)
    partMetrics = eval_by_parts(allParts, partPredDf)
    subset_acc = accDf["subset_acc"].mean()
    partMetrics["subset_acc"] = subset_acc
    avgTp = partMetrics["avgTp"]
    avgTn = partMetrics["avgTn"]
    avgAcc = partMetrics["avgAcc"]
    minTp = partMetrics["minTp"]
    minTn = partMetrics["minTn"]

    return subset_acc, avgTp, avgTn, minTp, minTn


def train_all_views():
    allViews = get_view_filename()
    for viewFilename in tqdm(allViews, desc="view"):
        fit(viewFilename)


targetFilename = "front_view_img_labels.csv"


def func2(trial):
    return fit(trial, targetFilename)


def tune():
    study_name = targetFilename.split(".")[0]  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    print(f"Study name : {storage_name}")
    study = optuna.create_study(
        study_name=study_name,
        directions=[
            "maximize",
            "maximize",
            "maximize",
        ],
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(func2, n_trials=100)


if __name__ == "__main__":
    # train_all_views()
    tune()
