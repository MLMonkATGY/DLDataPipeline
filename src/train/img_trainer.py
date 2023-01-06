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

import os
import ujson as json
from tqdm import tqdm
import numpy as np
import copy
from pytorch_lightning.loggers import MLFlowLogger
from src.analysis.ensemble_predictions import (
    ensemble_pred,
    eval_by_parts,
    get_raw_multilabel_df,
)
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
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import MultilabelPrecisionRecallCurve

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
from sklearn.metrics import hamming_loss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

wr.config.s3_endpoint_url = "http://localhost:8333"

np.random.seed(trainParams.randomSeed)
torch.manual_seed(trainParams.randomSeed)
warnings.filterwarnings("ignore")


class AsymmetricLoss(torch.nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


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


def create_model():

    # load Faster RCNN pre-trained model

    model = torchvision.models.efficientnet_b1(
        weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT
    )

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(
        in_features=num_ftrs, out_features=len(trainParams.targetPart)
    )

    return model


def get_dataloader(y_train, y_eval, y_test):
    trainTransform = A.Compose(
        [
            A.LongestMaxSize(trainParams.imgSize),
            A.PadIfNeeded(
                min_height=trainParams.imgSize,
                min_width=trainParams.imgSize,
                border_mode=0,
            ),
            A.ColorJitter(p=0.5),
            A.Rotate(limit=180, p=0.3, border_mode=0),
            # A.RandomGridShuffle(p=0.2),
            # A.CoarseDropout(max_height=16, max_width=16, p=0.2),
            # A.GaussianBlur(blur_limit=(1, 5), p=0.2),
            # A.Downscale(scale_min=0.6, scale_max=0.8, p=0.2),
            # A.GridDistortion(border_mode=0, p=0.2),
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
    evalDs = MultilabelDataset(y_eval, evalTransform)
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=trainParams.trainBatchSize,
        num_workers=trainParams.trainCPUWorker,
    )
    testDs = MultilabelDataset(y_test, evalTransform)
    testLoader = DataLoader2(
        testDs,
        shuffle=False,
        batch_size=trainParams.trainBatchSize,
        num_workers=trainParams.trainCPUWorker,
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


class ProcessModel(pl.LightningModule):
    def __init__(self, model, pos_weight=None):
        super(
            ProcessModel,
            self,
        ).__init__()
        pl.seed_everything(trainParams.randomSeed)

        self.model = model
        self.testAccMetric = MultilabelAccuracy(num_labels=len(trainParams.targetPart))
        self.trainAccMetric = MultilabelAccuracy(num_labels=len(trainParams.targetPart))
        self.testConfMat = MultilabelConfusionMatrix(
            num_labels=len(trainParams.targetPart), normalize="true"
        ).to(self.device)

        self.testF1Score = F1Score(
            task="multilabel", num_labels=len(trainParams.targetPart)
        ).to(self.device)
        self.trainConfMat = MultilabelConfusionMatrix(
            num_labels=len(trainParams.targetPart), normalize="true"
        ).to(self.device)

        self.testPrecision = Precision(
            task="multilabel",
            num_labels=len(trainParams.targetPart),
        ).to(self.device)
        self.testRecall = Recall(
            task="multilabel", num_labels=len(trainParams.targetPart)
        ).to(self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.current_pos_weight = pos_weight
        self.posThreshold = trainParams.posThreshold
        self.criterion = FocalLoss2d(weight=torch.tensor(self.current_pos_weight))
        self.pr_curve = MultilabelPrecisionRecallCurve(
            num_labels=len(trainParams.targetPart), thresholds=11
        )
        self.save_hyperparameters()

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
        self.trainAccMetric.reset()
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
        self.testF1Score.update(preds, targets.type(torch.int64))
        self.log("e_loss", loss, prog_bar=True)

        return preds, targets

    def validation_epoch_end(self, val_step_outputs) -> None:
        testAcc = self.testAccMetric.compute()
        testPrecision = self.testPrecision.compute()
        testRecall = self.testRecall.compute()
        testF1 = self.testF1Score.compute()
        preds = torch.cat([x[0] for x in val_step_outputs])
        targets = torch.cat([x[1] for x in val_step_outputs])

        exactAcc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        hammingLoss = hamming_loss(targets.cpu().numpy(), preds.cpu().numpy())
        hammingScore = 1 - hammingLoss
        self.log("e_acc", testAcc, prog_bar=False)
        self.log("precision", testPrecision, prog_bar=False)
        self.log("recall", testRecall, prog_bar=False)
        self.log("exact", exactAcc, prog_bar=False)
        self.log("subset", hammingScore, prog_bar=True)
        self.log("f1", testF1, prog_bar=True)

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
        self.testF1Score.reset()
        return super().validation_epoch_end(val_step_outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch["img"]
        files = batch["file"]
        parts = [x[0] for x in batch["parts"]]
        labels = batch["target"].cpu().numpy()
        logit = self(imgs)
        preds = (logit > self.posThreshold).cpu().numpy()
        targets = batch["target"].type(torch.uint8)
        self.pr_curve.update(logit, targets)

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


def train_eval(
    trainLoader, valLoader, testLoader, posWeight: List[float], kfoldId: int
):
    logger = MLFlowLogger(
        experiment_name=trainParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=f"{trainParams.imgAngle}_{kfoldId}",
    )
    trainParams.currentPosWeight = posWeight
    checkpoint_callback = ModelCheckpoint(
        monitor="subset",
        save_top_k=trainParams.saveTopNBest,
        mode="max",
        filename="{subset:.2f}-{e_tp:.2f}--{e_tn:.2f}",
    )
    model = create_model()

    trainProcessModel = ProcessModel(model, pos_weight=posWeight)
    trainer1 = pl.Trainer(
        accumulate_grad_batches=5,
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
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="f1", mode="max", patience=10),
        ],
        detect_anomaly=False,
        # limit_train_batches=1,
        # limit_val_batches=10,
        # limit_predict_batches=30,
    )

    trainer1.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    trainer1.predict(trainProcessModel, valLoader)
    precision, recall, threshold = trainProcessModel.pr_curve.compute()
    precision = precision[:, :-1]
    recall = recall[:, :-1]
    parts = trainParams.targetPart
    allPRByPart = pd.DataFrame()
    for p, r, part in zip(precision, recall, parts):
        prDf = pd.DataFrame(
            {"precision": p, "recall": r, "threshold": threshold, "part": part}
        )
        allPRByPart = pd.concat([allPRByPart, prDf])
    allPRByPart["f1"] = allPRByPart.apply(
        lambda x: 2
        * np.divide(x["precision"] * x["recall"], x["precision"] + x["recall"]),
        axis=1,
    )
    allPRByPart["view"] = trainParams.imgAngle
    trainProcessModel.pr_curve.reset()
    testDf = trainer1.predict(trainProcessModel, testLoader)
    completePredDf = pd.concat(testDf)

    displayLogger.success("Started Uploading Best Checkpoints..")
    mlflowLogger: MlflowClient = trainProcessModel.logger.experiment
    mlflowLogger.log_dict(
        trainProcessModel.logger.run_id,
        dataclasses.asdict(trainParams),
        "hyperparams.json",
    )
    outputThresholdCsv = os.path.join(checkpoint_callback.dirpath, "PR_thresholds.csv")
    allPRByPart.to_csv(outputThresholdCsv)
    mlflowLogger.log_artifacts(
        trainProcessModel.logger.run_id,
        checkpoint_callback.dirpath.replace("/checkpoints", "/"),
    )
    return completePredDf, trainProcessModel.logger.experiment_id, allPRByPart


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


def store_pred(completePred: pd.DataFrame, expId: int):
    run_name = f"cv_pred_{trainParams.imgAngle}"
    outputDir = pathlib.Path(os.getcwd()) / "outputs"
    os.makedirs(outputDir, exist_ok=True)
    with mlflow.start_run(experiment_id=expId, run_name=run_name):
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
    base = [
        "rear_view_img_labels.csv",
        "rear_view_left_img_labels.csv",
        "rear_view_right_img_labels.csv",
        "front_view_left_img_labels.csv",
        "front_view_img_labels.csv",
        "front_view_right_img_labels.csv",
    ]
    remoteFilename = [f"{trainParams.vehicleType}_{x}" for x in base]
    return remoteFilename


def get_label_df(filename):
    labelDf = wr.s3.read_csv(path=f"s3://imgs_labels_3/{filename}")

    return labelDf


def gen_dataset(viewFilename):
    labelDf: pd.DataFrame = get_label_df(viewFilename)
    testCaseId = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/test_set/test_case_id.csv"
    )["CaseID"].tolist()
    testLabelDf = labelDf[labelDf["CaseID"].isin(testCaseId)]
    labelDf = labelDf[~labelDf["CaseID"].isin(testCaseId)].reset_index()
    logger.success(f"Test size : {len(testLabelDf)}")
    allPartsFound = [x for x in labelDf.filter(regex="vision_*").columns]
    allVisionFeatures = [
        "vision_bonnet",
        "vision_bumper_front",
        "vision_engine",
        "vision_grille",
        "vision_headlamp_lh",
        "vision_headlamp_rh",
        "vision_bumper_rear",
        "vision_front_panel",
        "vision_non_external",
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
    allTargetParts = sorted(list(set(allVisionFeatures) & set(allPartsFound)))
    trainParams.runName = viewFilename
    trainParams.targetPart = allTargetParts
    trainParams.imgAngle = viewFilename.replace("_img_labels.csv", "")

    y = labelDf[allTargetParts]
    X = labelDf["filename"]
    test_y = testLabelDf[allTargetParts]
    test_X = testLabelDf["filename"]
    return X, y, test_X, test_y


def get_pos_weight(trial, allColName, viewFilename) -> List[float]:
    if trial:
        posWeight = [
            trial.suggest_float(f"pos_weight_{allColName[x]}", 0.05, 20, log=True)
            for x in range(len(allColName))
        ]
    else:
        study_name = viewFilename.split(".")[0]
        baseDir = (
            "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/hyperparams"
        )
        dbFile = f"sqlite:///{baseDir}/{trainParams.vehicleType}_{study_name}.db"
        study = optuna.load_study(study_name=study_name, storage=dbFile)
        df: pd.DataFrame = study.trials_dataframe(
            attrs=("number", "value", "params", "state")
        )
        majorMetricsInOrder = ["values_1", "values_2", "values_0", "values_3"]
        df.sort_values(by=majorMetricsInOrder, ascending=False, inplace=True)
        print(df[majorMetricsInOrder])
        colData = [f"params_pos_weight_{x}" for x in allColName]
        print(df[colData])
        posWeight = df[colData].head(1).values.tolist()[0]
        return posWeight


def select_best_threshold(df: pd.DataFrame):
    df.sort_values(by="f1", ascending=False, inplace=True)
    allBestThreshold = df.groupby("part").head(1)
    # """
    # TODO
    # """
    # allBestThreshold["threshold"] = 0.5
    return allBestThreshold


def fit(viewFilename):

    X, y, test_X, test_y = gen_dataset(viewFilename)
    mulitlearnStratify = MultilabelStratifiedKFold(
        n_splits=trainParams.kFoldSplit,
        shuffle=True,
        random_state=trainParams.randomSeed,
    )
    allColName = y.columns.tolist()
    trainParams.currentColName = allColName
    viewCompletePredDf = pd.DataFrame()
    viewCompletePRThreshold = pd.DataFrame()

    for kfoldId, (train_index, test_index) in enumerate(mulitlearnStratify.split(X, y)):
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y.loc[train_index], y.loc[test_index]
        y_train["filename"] = X_train
        y_eval["filename"] = X_eval
        test_y["filename"] = test_X

        trainLoader, valLoader, testLoader = get_dataloader(y_train, y_eval, test_y)
        posWeight = trainLoader.dataset.allPosWeight
        predDf, expId, allPRByPart = train_eval(
            trainLoader, valLoader, testLoader, posWeight, kfoldId + 1
        )
        viewCompletePRThreshold = pd.concat([viewCompletePRThreshold, allPRByPart])
        viewCompletePredDf = pd.concat([viewCompletePredDf, predDf])
        """
        #TODO : No more cross validations
        """
        break
    bestThresholdDf = select_best_threshold(viewCompletePRThreshold)
    viewCompletePredDf = viewCompletePredDf.merge(
        bestThresholdDf, left_on="parts", right_on="part"
    )
    store_pred(viewCompletePredDf, expId)


def train_all_views():
    allViews = get_view_filename()

    for viewFilename in tqdm(allViews, desc="view"):
        fit(viewFilename)


if __name__ == "__main__":
    # allVehicleType = ["Saloon - 4 Dr", "Hatchback - 5 Dr", "SUV - 5 Dr"]
    allVehicleType = ["SUV - 5 Dr"]

    for vehicleType in allVehicleType:
        trainParams.vehicleType = vehicleType
        train_all_views()
