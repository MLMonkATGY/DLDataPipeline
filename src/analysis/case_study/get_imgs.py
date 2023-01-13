import glob
import pandas as pd
import os
import shutil
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

rows = 2
columns = 3

plt.ioff()


def plot(fig, imgNp, viewName, idx):
    fig.tight_layout()
    fig.add_subplot(rows, columns, idx)

    # showing image
    plt.imshow(imgNp)
    plt.axis("off")

    plt.title(viewName)


src = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs_metadata/**.parquet"
allCsv = glob.glob(src, recursive=True)
caseStudyId = pd.read_csv(
    "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/case_study/case_study_16/case_study_gt.csv"
)["CaseID"].tolist()
imgSrc = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/imgs"

allDf = []
for i in allCsv:
    allDf.append(pd.read_parquet(i))
localFileDf = pd.concat(allDf)
caseStudyFilesDf = localFileDf[localFileDf["CaseID"].isin(caseStudyId)]
print(caseStudyFilesDf.columns)
caseStudyFilesDf.sort_values(by="StdDocDesc", inplace=True)
caseStudyFilesDf["filepath"] = caseStudyFilesDf["filename"].apply(
    lambda x: os.path.join(imgSrc, x)
)
print(caseStudyFilesDf[["CaseID", "filename", "filepath", "StdDocDesc"]])
imgSink = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/case_study/case_study_16/imgs"
os.makedirs(imgSink, exist_ok=True)
for caseId in tqdm(caseStudyFilesDf["CaseID"].unique()):
    allViewImg = caseStudyFilesDf[caseStudyFilesDf["CaseID"] == caseId]
    fig = plt.figure(figsize=(16, 12))
    idx = 1
    plt.axis("off")
    plt.title(f"CaseID {caseId}", y=0.95)
    for _, i in allViewImg.iterrows():
        imgNp = cv2.imread(i["filepath"])
        imgNp = cv2.cvtColor(imgNp, cv2.COLOR_BGR2RGB)
        viewName = i["StdDocDesc"]
        plot(fig, imgNp, viewName, idx)
        idx += 1
    gridFile = f"{imgSink}/{caseId}.png"

    plt.savefig(gridFile)
    plt.close("all")
