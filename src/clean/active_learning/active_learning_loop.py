import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader2
from torchvision import transforms, utils
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from src.TrainClassifierParams import trainParams
import glob
import shutil
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends
import torch.utils.data as torchdata
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision import models
from torchvision.transforms import transforms
from tqdm.autonotebook import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import ModelWrapper
from flash.image.classification.integrations.baal import (
    ActiveLearningDataModule,
    ActiveLearningLoop,
)


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        targetName: str,
        transform=None,
    ):
        self.df: pd.DataFrame = df
        self.srcImgDir = img_dir
        self.targetName = targetName
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        targetRow = self.df.iloc[idx]
        # print(targetRow)
        targetFilePath = os.path.join(self.srcImgDir, targetRow["filename"])
        label = targetRow[self.targetName]
        labelTensor = torch.tensor(label, dtype=torch.long)
        image = cv2.imread(targetFilePath)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)
        # sample["image"] = sample["image"] / 255.0

        data = {
            "img": sample["image"],
            "target": labelTensor,
            "file": targetFilePath,
            "filename": targetRow["filename"],
        }
        return data


def select_first_batch(df: pd.DataFrame):
    df.sort_values(by="aum", ascending=False, inplace=True)
    firstBatchDf = df.groupby(["view", "gt"]).head(200)
    print(len(firstBatchDf))
    return firstBatchDf


def transport_files(df: pd.DataFrame, partName: str, version: int):
    outputBaseDir = (
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/al"
    )
    srcimgDir = (
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"
    )
    partDir = f"{outputBaseDir}/{partName}/v{version}"
    os.makedirs(partDir, exist_ok=True)
    dmgDir = f"{partDir}/dmg"
    os.makedirs(dmgDir, exist_ok=True)
    notdmgDir = f"{partDir}/not_dmg"
    os.makedirs(notdmgDir, exist_ok=True)
    notDmgDfSample = df[df["gt"] == 0]["filename"]
    dmgDfSample = df[df["gt"] == 1]["filename"]
    for dmgFile in dmgDfSample:
        fullPath = f"{srcimgDir}/{dmgFile}"
        shutil.copy(fullPath, dmgDir)
    for notDmgFile in notDmgDfSample:
        fullPath = f"{srcimgDir}/{notDmgFile}"
        shutil.copy(fullPath, notdmgDir)


def get_datasets(initial_pool):
    """
    Let's create a subset of CIFAR10 named CIFAR3, so that we can visualize thing better.

    We will only select the classes airplane, cat and dog.

    Args:
        initial_pool: Amount of labels to start with.

    Returns:
        ActiveLearningDataset, Dataset, the training and test set.
    """

    class TransformAdapter(torchdata.Subset):
        # We need a custom Subset class as we need to override "transforms" as well.
        # This shouldn't be needed for your experiments.
        @property
        def transform(self):
            if hasattr(self.dataset, "transform"):
                return self.dataset.transform
            else:
                raise AttributeError()

        @transform.setter
        def transform(self, transform):
            if hasattr(self.dataset, "transform"):
                self.dataset.transform = transform

    # airplane, cat, dog
    classes_to_keep = [0, 3, 5]
    transform = transforms.Compose(
        [
            transforms.Resize((480, 480)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            # transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )

    train_mask = np.where([y in classes_to_keep for y in train_ds.targets])[0]
    train_ds = TransformAdapter(train_ds, train_mask)

    # In a real application, you will want a validation set here.
    test_set = datasets.CIFAR10(
        ".", train=False, transform=test_transform, target_transform=None, download=True
    )
    test_mask = np.where([y in classes_to_keep for y in test_set.targets])[0]
    test_set = TransformAdapter(test_set, test_mask)

    # Here we set `pool_specifics`, where we set the transform attribute for the pool.
    active_set = ActiveLearningDataset(
        train_ds, pool_specifics={"transform": test_transform}
    )

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set


def init():
    allParts = [
        "vision_bumper_rear",
        "vision_fender_front_lh",
        "vision_fender_front_rh",
        "vision_headlamp_lh",
        "vision_headlamp_rh",
        "vision_grille",
        "vision_windscreen_front",
        "vision_windscreen_rear",
        "vision_tail_lamp_lh",
        "vision_tail_lamp_rh",
        "vision_rear_quarter_rh",
        "vision_rear_quarter_lh",
        "vision_bonnet",
        "vision_bumper_front",
        "vision_engine",
        "vision_front_panel",
        "vision_rear_compartment",
        "vision_rear_panel",
    ]
    allParts = [x.replace("vision_", "") for x in allParts]
    srcDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/aum_v2"
    for p in allParts:
        searchTerm = f"{p}_**_clean"
        partDirs = glob.glob(f"{srcDir}/{searchTerm}", recursive=True)
        allDf = []
        for i in partDirs:
            infoDfCsv = f"{i}/all_info_after_select.csv"
            viewName = (
                i.split("/")[-1].replace(f"{p}_", "").replace("_img_labels_clean", "")
            )
            infoDf = pd.read_csv(infoDfCsv)
            infoDf["view"] = viewName
            allDf.append(infoDf)
        partDf = pd.concat(allDf)
        firstBatchDf = select_first_batch(partDf)
        transport_files(firstBatchDf, p, 1)


if __name__ == "__main__":
    init()
