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
        data = {
            "img": sample["image"],
            "target": labelTensor,
            "file": targetFilePath,
            "filename": targetRow["filename"],
        }
        return data


class PredictionImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        targetName: str,
        targetView: str,
        transform=None,
    ):
        self.df: pd.DataFrame = df
        self.srcImgDir = img_dir
        self.targetName = targetName
        self.transform = transform
        self.targetView = targetView

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        targetRow = self.df.iloc[idx]
        target = torch.tensor(targetRow[self.targetName])
        targetFilePath = os.path.join(self.srcImgDir, targetRow["filename"])
        image = cv2.imread(targetFilePath)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)
        data = {
            "img": sample["image"],
            "target": target,
            "file": targetFilePath,
            "filename": targetRow["filename"],
            "vehicleType": targetRow["Vehicle_Type"],
            "model": targetRow["Model"],
        }
        return data


if __name__ == "__main__":
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
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    ds = ImageDataset(
        df_path="/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/windscreen_rear_Rear View_img_labels.csv",
        img_dir="/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs",
        transform=trainTransform,
        targetName="vision_windscreen_rear",
    )
    loader = DataLoader2(
        ds,
        batch_size=100,
        shuffle=False,
        num_workers=10,
    )
    for i in loader:
        print(i)
