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
from TrainClassifierParams import trainParams


class DmgClassifierDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, colName, transform=None, targetFile=[]):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        with open(trainParams.rejectFile, "r") as f:
            allRejFile = f.read().split("\n")

        if len(targetFile) > 0:
            self.df = self.df[self.df["Path"].isin(targetFile)]
            beforeRemove = len(self.df)
            self.df = self.df[~self.df["Path"].isin(allRejFile)]
            afterRemove = len(self.df)
            print(f"Removed from reject file : {beforeRemove - afterRemove}")
        self.colName = colName
        # assert self.colName in csv_file
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        targetRow = self.df.iloc[idx]
        targetFilePath = targetRow["Path"]
        label = targetRow[self.colName]
        labelTensor = torch.tensor(label, dtype=torch.int64)
        image = cv2.imread(targetFilePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)
        data = {"img": sample["image"], "target": labelTensor, "file": targetFilePath}
        return data


if __name__ == "__main__":

    trainCsv = "/home/alextay96/Desktop/new_workspace/partlist_prediction/data/interim/label_issue/clean/front_fender_LH_v1_Saloon-4Dr_Front View Left_img_label.csv"
    colName = "front_fender_LH"
    trainTransform = A.Compose(
        [
            A.LongestMaxSize(600),
            A.PadIfNeeded(min_height=600, min_width=600),
            ToTensorV2(),
        ]
    )
    ds = DmgClassifierDataset(trainCsv, colName, trainTransform)
    loader = DataLoader2(ds, batch_size=5, shuffle=True)
    for img, labels in loader:
        print(labels)
