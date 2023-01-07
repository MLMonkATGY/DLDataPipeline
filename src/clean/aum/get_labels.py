from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import plotly.express as px
import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt
import awswrangler as wr
import boto3
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import os
import random
from joblib import Parallel, delayed
from pprint import pprint
from pathlib import Path
import ujson as json


def get_labels(partName):
    targetPart = partName.replace("vision_", "")

    wr.config.s3_endpoint_url = "http://192.168.1.4:8333"
    vehicleType = ["Saloon - 4 Dr", "Hatchback - 5 Dr", "SUV - 5 Dr"]

    def myFilter(x):
        return x["Vehicle_Type"] in vehicleType

    baseOutputDir = Path(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset"
    )
    filesDf = wr.s3.read_parquet(
        f"s3://scope_file/", dataset=True, columns=["CaseID", "iDOCID"]
    )
    print(filesDf.columns)
    labelDf = wr.s3.read_parquet(path=f"s3://multilabel_df/", dataset=True)
    print(labelDf.columns)
    endpoint = "http://192.168.1.4:8888/buckets/raw_imgs/"

    mappingDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/tmp/complete_view_mapping.csv"
    )
    mappingDf["lvl_3_desc"] = mappingDf["lvl_3_desc"].str.strip()
    availableViews = [
        "Front View",
        "Rear View",
        "Front View Left",
        "Front View Right",
        "Rear View Left",
        "Rear View Right",
    ]
    relatedViews = []
    for v in availableViews:
        targetRow = mappingDf[
            mappingDf["lvl_3_desc"] == targetPart.replace("_", " ").lower()
        ]
        if len(targetRow[targetRow[v.replace(" ", "_").lower()] == 1]) > 0:
            relatedViews.append(v)

    caseDf = wr.s3.read_parquet(
        f"s3://scope_case/",
        partition_filter=myFilter,
        dataset=True,
        columns=["CaseID", "Vehicle_Type"],
    )

    cli = boto3.client(
        "s3",
        **{
            "endpoint_url": "http://192.168.1.4:8333",
        },
    )
    paginator = cli.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": "raw_imgs"}
    page_iterator = paginator.paginate(**operation_parameters)
    # downloadedImgs = []
    # for page in tqdm(page_iterator):
    #     pageContent = page["Contents"]
    #     downloadedCaseId = set([int(x["Key"].split("_")[0]) for x in pageContent])
    #     downloadedImgs.extend(downloadedCaseId)
    # with open("./downloaded_case_id.json", "w") as f:
    #     json.dump(downloadedImgs, f)
    with open("./downloaded_case_id.json", "r") as f:
        downloadedImgs = json.load(f)
    print(len(downloadedImgs))
    availableImgsCase = set(downloadedImgs).intersection(
        caseDf["CaseID"].unique().tolist()
    )
    downloadedCaseLength = len(availableImgsCase)
    filesDf = filesDf[filesDf["CaseID"].isin(availableImgsCase)]
    balanceLabelDf = labelDf[
        labelDf["CaseID"].isin(filesDf["CaseID"].unique().tolist())
    ]
    balanceLabelDf = balanceLabelDf.sample(frac=1).groupby(targetPart).head(15000)
    print(balanceLabelDf[targetPart].value_counts())

    viewToPart = dict()
    for v in relatedViews:
        partsInView = [targetPart]
        viewToPart[v] = ["vision_" + x.replace(" ", "_") for x in partsInView]

    viewDfMap = dict()
    print(filesDf["StdDocDesc"].value_counts())
    for v, parts in viewToPart.items():
        tempDf = balanceLabelDf[parts + ["CaseID"]]
        tempDf["view"] = v
        viewFilesDf = filesDf[filesDf["StdDocDesc"] == v]
        viewFilesDf.drop_duplicates(subset=["CaseID"], inplace=True)
        viewFilesDf["filename"] = viewFilesDf[["CaseID", "iDOCID"]].apply(
            lambda x: str(int(x["CaseID"])) + "_" + str(int(x["iDOCID"])) + ".JPG",
            axis=1,
        )
        viewFilesDf["url"] = endpoint + viewFilesDf["filename"]
        tempDf = tempDf.merge(viewFilesDf[["filename", "CaseID", "url"]], on="CaseID")
        viewDfMap[v] = tempDf
    for view, df in viewDfMap.items():
        imgLabelFilename = f"{targetPart}_{view}_img_labels.csv"
        outputLabelFile = baseOutputDir / imgLabelFilename
        df.to_csv(outputLabelFile)
        # wr.s3.to_csv(
        #     df=df,
        # path=f"s3://imgs_labels_4/{imgLabelFilename}",

        # )


if __name__ == "__main__":

    get_labels("vision_")
