import pandas as pd
import glob
import numpy as np

srcDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs_metadata"
allDf = []
for i in glob.glob(f"{srcDir}/**.parquet", recursive=True):

    df = pd.read_parquet(i)
    allDf.append(df)
completeFileDf = pd.concat(allDf)
allCaseId = completeFileDf["CaseID"].unique().tolist()
np.random.shuffle(allCaseId)
testSetSize = 5000

testCaseId = pd.DataFrame({
    "CaseID" : allCaseId[:testSetSize]
})
testCaseId.to_csv("./test_case_id.csv")
