import glob
import pandas as pd
import os
import shutil

src = (
    "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/imgs_metadata/**.parquet"
)
allCsv = glob.glob(src, recursive=True)
caseStudyId = pd.read_csv(
    "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/case_study/case_study_gt.csv"
)["CaseID"].tolist()
allDf = []
for i in allCsv:
    allDf.append(pd.read_parquet(i))
localFileDf = pd.concat(allDf)
caseStudyFilesDf = localFileDf[localFileDf["CaseID"].isin(caseStudyId)]
imgSrc = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/imgs"
allFilePath = [os.path.join(imgSrc, x) for x in caseStudyFilesDf["filename"].tolist()]
imgSink = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/case_study/imgs"
for i in allFilePath:
    shutil.copy(i, imgSink)
