import mlflow
import ujson as json
from mlflow.tracking import MlflowClient
from ClassifierDataset import DmgClassifierDataset
from KFoldTrainPredict import ProcessModel
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


def GetCrossValPred():
    with open("./data/angle.json", "r") as f:
        anglePartMap = json.load(f)
    outputDir = pathlib.Path("./cross_val_pred")
    allMetrics = []
    for view, parts in tqdm(anglePartMap.items(), desc="angle"):
        for part in tqdm(parts, desc="part"):
            for k in range(1, 6):

                runName = f"{view}_{part}_{k}"
                query = f"tags.`mlflow.runName`='{runName}'"
                runs = MlflowClient().search_runs(
                    experiment_ids=["65"],
                    filter_string=query,
                    order_by=["attribute.start_time DESC"],
                    max_results=1,
                )
                runs[0].data.metrics["kfold"] = k
                runs[0].data.metrics["part"] = part
                runs[0].data.metrics["view"] = view

                allMetrics.append(runs[0].data.metrics)
    evalDf = pd.json_normalize(allMetrics)
    evalDf.to_csv("./tmp/models_43_perf.csv")
    print(evalDf)
    # mlflow.artifacts.download_artifacts(run_id=runId, dst_path=outputModelDir)


if __name__ == "__main__":
    GetCrossValPred()
