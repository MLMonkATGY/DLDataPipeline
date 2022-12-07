import pandas as pd

pd.set_option("display.max_rows", 300)
import rapidfuzz
from joblib import Parallel, delayed
from tqdm import tqdm
import re
import awswrangler as wr
import boto3
import copy
import itertools


def worker(caseIdList, labelDf, rawMapping):
    allPayload = []
    labelDf = labelDf[labelDf["CaseID"].isin(caseIdList)]
    for caseId in tqdm(caseIdList, desc="case"):
        targets = labelDf[labelDf["CaseID"] == caseId]

        caseData = copy.deepcopy(rawMapping)
        caseData["CaseID"] = targets.iloc[0]["CaseID"]
        caseData["Circumstances_of_Accident"] = targets.iloc[0][
            "Circumstances_of_Accident"
        ]
        lvl2DmgParts = targets["lvl_2_desc"].tolist()
        lvl3DmgParts = targets["lvl_3_desc"].tolist()

        for part in lvl3DmgParts:
            caseData[f"vision_{part}"] = 1
        for part in lvl2DmgParts:
            caseData[part] = 1

        allPayload.append(caseData)
    return pd.json_normalize(allPayload)


if __name__ == "__main__":
    wr.config.s3_endpoint_url = "http://192.168.1.7:8333"

    labelDf = wr.s3.read_parquet(
        path=f"s3://partlist_label/",
        columns=["CaseID", "lvl_1_desc", "lvl_2_desc", "lvl_3_desc"],
    )
    caseDf = wr.s3.read_parquet(
        path=f"s3://scope_case/",
        # chunked=100
    )
    labelDf = labelDf.merge(caseDf, on="CaseID")
    allLvl3Labels = labelDf["lvl_3_desc"].unique().tolist()
    allLvl2Labels = labelDf["lvl_2_desc"].unique().tolist()

    rawMapping = {f"vision_{x}": 0 for x in allLvl3Labels}
    for x in allLvl2Labels:
        rawMapping[x] = 0
    allPayload = []
    allDf = []
    allCaseId = labelDf["CaseID"].unique().tolist()
    tasks = []
    batchSize = 10000
    worker_num = 8
    sampleBatch = 40
    for i in range(0, len(allCaseId), batchSize):
        tasks.append(allCaseId[i : i + batchSize])
    result = Parallel(n_jobs=3)(
        delayed(worker)(task, labelDf, rawMapping) for task in tqdm(tasks)
    )
    multilabelDf = pd.concat(result)
    print(multilabelDf)
    cli = boto3.client(
        "s3",
        **{
            "endpoint_url": "http://192.168.1.7:8333",
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            # "Username": "aaa",
        },
    )
    outputBucketName = "multilabel_df"
    # cli.create_bucket(Bucket=outputBucketName)
    wr.s3.to_parquet(
        df=multilabelDf,
        path=f"s3://{outputBucketName}/",
        dataset=True,
        mode="overwrite",
    )
