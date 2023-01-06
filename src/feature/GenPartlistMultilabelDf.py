import itertools
import copy
import boto3
import awswrangler as wr
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import rapidfuzz
import pandas as pd

pd.set_option("display.max_rows", 300)


def worker(caseIdList, labelDf, rawMapping):
    allPayload = []
    structuredFeatures = [
        "Circumstances_of_Accident",
        "Model",
        "Assembly_Type",
        "Vehicle_Still_Driveable",
        "NCB_Stat",
        "Assembly_Type",
        "Claim_Type",
        "Vehicle_Type",
        "Sum_Insured",
        "Repairer",
        "Repairer_Apprv_Count",
        "Collision_With",
        "Handling_Insurer",
    ]
    labelDf = labelDf[labelDf["CaseID"].isin(caseIdList)]
    for caseId in tqdm(caseIdList, desc="case"):
        targets = labelDf[labelDf["CaseID"] == caseId]

        caseData = copy.deepcopy(rawMapping)
        caseData["CaseID"] = targets.iloc[0]["CaseID"]
        for col in structuredFeatures:
            caseData[col] = targets.iloc[0][col]
        lvl2DmgParts = targets["lvl_2_desc"].tolist()
        lvl3DmgParts = targets["lvl_3_desc"].tolist()

        for part in lvl3DmgParts:
            caseData[f"vision_{part}"] = 1
        for part in lvl2DmgParts:
            caseData[part] = 1

        allPayload.append(caseData)
    return pd.json_normalize(allPayload)


if __name__ == "__main__":
    # noisyLabelDf = pd.read_csv(
    #     "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/cleanlab/v3/noisy_label_top15.csv"
    # )
    # noisyLabelCase = noisyLabelDf["CaseID"].tolist()
    wr.config.s3_endpoint_url = "http://192.168.1.4:8333"

    labelDf = wr.s3.read_parquet(
        path=f"s3://partlist_label/",
        columns=["CaseID", "lvl_1_desc", "lvl_2_desc", "lvl_3_desc"],
    )
    caseDf = wr.s3.read_parquet(
        path=f"s3://scope_case/",
        dataset=True
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
    validCaseId = allCaseId
    # print(len(validCaseId))
    tasks = []
    batchSize = 10000
    worker_num = 3
    sampleBatch = 40
    for i in range(0, len(validCaseId), batchSize):
        tasks.append(validCaseId[i: i + batchSize])
    result = Parallel(n_jobs=worker_num)(
        delayed(worker)(task, labelDf, rawMapping) for task in tqdm(tasks)
    )
    multilabelDf = pd.concat(result)
    print(multilabelDf)
    cli = boto3.client(
        "s3",
        **{
            "endpoint_url": "http://192.168.1.4:8333",
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            # "Username": "aaa",
        },
    )
    outputBucketName = "multilabel_df_3"
    cli.create_bucket(Bucket=outputBucketName)
    wr.s3.to_parquet(
        df=multilabelDf,
        path=f"s3://{outputBucketName}/",
        dataset=True,
        mode="overwrite",
    )
