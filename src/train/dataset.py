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


class MultilabelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
    ):
        self.df: pd.DataFrame = df
        self.colName = [x for x in self.df.filter(regex="vision_*").columns]
        # assert self.colName in csv_file
        self.transform = transform
        self.allPosWeight = []
        self.srcImgDir = trainParams.srcImgDir
        for col in self.colName:
            if len(self.df[self.df[col] == 0]) > 0:
                posSampleSize = len(self.df[self.df[col] == 1])
                negSampleSize = len(self.df[self.df[col] == 0])
                posWeight = negSampleSize / posSampleSize

            else:
                posWeight = 1
            self.allPosWeight.append(posWeight)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        targetRow = self.df.iloc[idx]

        targetFilePath = os.path.join(self.srcImgDir, targetRow["filename"])
        if not os.path.exists(targetFilePath):
            print(targetFilePath)
        label = targetRow[self.colName]
        labelTensor = torch.tensor(label, dtype=torch.float32)
        image = cv2.imread(targetFilePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)
        data = {
            "img": sample["image"],
            "target": labelTensor,
            "file": targetFilePath,
            "parts": self.colName,
        }
        return data
