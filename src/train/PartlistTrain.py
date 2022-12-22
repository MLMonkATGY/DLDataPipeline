from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
)
import numpy as np
import plotly.express as px
import shap
import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt
import awswrangler as wr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    multilabel_confusion_matrix,
)
from catboost.utils import select_threshold
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
from catboost.utils import get_roc_curve, select_threshold
from torchmetrics.classification import (
    MulticlassPrecisionRecallCurve,
    BinaryPrecisionRecallCurve,
)
import torch


def train_catboost():
    wr.config.s3_endpoint_url = "http://192.168.1.4:8333"

    multilabelDf = wr.s3.read_parquet(
        path=f"s3://multilabel_df/",
        dataset=True,
    )
    allVisionFeatures = [x for x in multilabelDf.columns if "vision_" in x]
    caseFeatures = [
        "Circumstances_of_Accident",
    ]
    allInputFeature = caseFeatures + allVisionFeatures
    targetCol = [
        x for x in multilabelDf.columns if x not in allInputFeature and x != "CaseID"
    ]
    saloonPredDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/results/saloon_focal_480_aug_2/Saloon - 4 Dr_imgs_pred_output.csv"
    )
    # hatchBackDf = pd.read_csv("/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results/hatchback_v2/Hatchback - 5 Dr_imgs_pred_output.csv")
    # suvDf = pd.read_csv("/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/results/suv_v2/SUV - 5 Dr_imgs_pred_output.csv")

    # imgPredOutput = pd.concat([saloonPredDf, hatchBackDf, suvDf])
    imgPredOutput = saloonPredDf
    allPredModel = []
    allPreds = []
    allGt = []
    realTestDataDf = multilabelDf[caseFeatures + targetCol + ["CaseID"]].merge(
        imgPredOutput, on="CaseID"
    )
    realTestDataDf = realTestDataDf.loc[
        :, ~realTestDataDf.columns.str.contains("^Unnamed")
    ]
    trainDf = multilabelDf[
        ~multilabelDf["CaseID"].isin(realTestDataDf["CaseID"].unique().tolist())
    ]
    assert set(realTestDataDf["CaseID"].tolist()).isdisjoint(trainDf["CaseID"].tolist())

    for part in tqdm(targetCol):
        allCaseIdByPart = []
        allPredByPart = []
        allGtByPart = []
        trainCaseId = trainDf["CaseID"].tolist()
        testCaseId = realTestDataDf["CaseID"].tolist()
        X_train = trainDf[allInputFeature]
        Y_train = trainDf[part].to_frame()
        X_test = realTestDataDf[allInputFeature]
        Y_test = realTestDataDf[part].to_frame()
        pos_count = len(Y_test[Y_test[part] == 1]) / len(Y_test)
        neg_count = len(Y_test[Y_test[part] == 0]) / len(Y_test)
        pos_weight = neg_count / pos_count
        train_pool = Pool(
            X_train, Y_train, cat_features=caseFeatures + allVisionFeatures
        )
        test_pool = Pool(X_test, Y_test, cat_features=caseFeatures + allVisionFeatures)
        prCurve = MulticlassPrecisionRecallCurve(num_classes=2, thresholds=11)
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            iterations=100,
            task_type="GPU",
            # scale_pos_weight=pos_weight,
            use_best_model=True
            # auto_class_weights="Balanced"
            # class_names=["not_dmg", "dmg"],
        )
        clf.fit(train_pool, eval_set=test_pool, metric_period=20, plot=False, verbose=0)
        test_predict = clf.predict(X_test)

        (fpr, tpr, thresholds) = get_roc_curve(clf, test_pool, plot=True)
        boundary = select_threshold(clf, curve=(fpr, tpr, thresholds), FPR=0.2)
        clf.set_probability_threshold(boundary)
        test_predict = clf.predict(X_test)

        acc = accuracy_score(Y_test.values.astype(np.int64), test_predict)
        confMat = confusion_matrix(Y_test.values.astype(np.int64), test_predict)
        pos_count = len(Y_test[Y_test[part] == 1]) / len(Y_test)
        tn = confMat[0][0]
        tp = confMat[1][1]
        fp = confMat[0][1]
        fn = confMat[1][0]
        totalSample = fp + fn + tp + tn
        acc = (tp + tn) / (fp + fn + tp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        assert (tp / (tp + fn)) + (fn / (tp + fn)) == (tn / (tn + fp)) + (
            fp / (tn + fp)
        )

        allPredModel.append(
            {
                "part": part,
                "tp": tp / (tp + fn),
                "tn": tn / (tn + fp),
                "fp": fp / (tn + fp),
                "fn": fn / (tp + fn),
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": (2 * precision * recall) / (precision + recall),
                "pos_count": pos_count,
            }
        )
        assert len(testCaseId) == len(test_predict)
        assert len(testCaseId) == len(Y_test)

        allPreds.append({"CaseID": testCaseId, part: test_predict})
        allGt.append({"CaseID": testCaseId, part: Y_test.values.squeeze(1)})

    evalMetrics = pd.json_normalize(allPredModel)
    avgPrecision = evalMetrics["precision"].mean()
    avgRecall = evalMetrics["recall"].mean()
    avgF1 = evalMetrics["f1"].mean()
    avgTp = evalMetrics["tp"].mean()
    avgTn = evalMetrics["tn"].mean()
    avgAcc = evalMetrics["acc"].mean()
    avgFn = evalMetrics["fn"].mean()

    print(f"Avg Precision : {avgPrecision}")
    print(f"Avg Recall : {avgRecall}")
    print(f"Avg F1 : {avgF1}")
    print(f"Avg TP : {avgTp}")
    print(f"Avg TN : {avgTn}")
    print(f"Avg FN : {avgFn}")

    print(f"avgAcc : {avgAcc}")


if __name__ == "__main__":
    train_catboost()
