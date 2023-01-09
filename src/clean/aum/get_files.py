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


def transport_worker(df: pd.DataFrame, sinkDir: str):
    s = requests.Session()
    for _, i in tqdm(df.iterrows(), desc="files"):
        url = i["url"]
        imgBytes = s.get(url)
        localPath = os.path.join(sinkDir, i["filename"])
        with open(localPath, "wb") as f:
            f.write(imgBytes.content)


def get_files(labelFile):
    sinkDir = (
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"
    )
    # labelFile = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/windscreen_front_Front View Left_img_labels.csv"
    downloadFileDf = pd.read_csv(labelFile)
    os.makedirs(sinkDir, exist_ok=True)
    s = requests.Session()
    batchTask = []
    batchSize = 1000
    for i in range(0, len(downloadFileDf), batchSize):
        batchTask.append(downloadFileDf.iloc[i : i + batchSize])
    Parallel(n_jobs=10)(
        delayed(transport_worker)(taskDf, sinkDir)
        for taskDf in tqdm(batchTask, desc="tasks")
    )
