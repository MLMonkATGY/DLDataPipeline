import pandas as pd
import boto3
from tqdm import tqdm
import ujson as json
import awswrangler as wr
from loguru import logger

wr.config.s3_endpoint_url = "http://localhost:8333"
kwrgs = {
    "endpoint_url": "http://localhost:8333",
    "aws_access_key_id": "",
    "aws_secret_access_key": "",
    # "Username": "aaa",
}
bucketName = "mrm_raw"
sinkBucketName = "raw_df"
cli = boto3.client("s3", **kwrgs)
caseBucket = "tmp_case"
partBucket = "tmp_part"
fileBucket = "tmp_file"
cli.create_bucket(Bucket=caseBucket)
cli.create_bucket(Bucket=partBucket)
cli.create_bucket(Bucket=fileBucket)

buckets = cli.list_buckets()
paginator = cli.get_paginator("list_objects_v2")
operation_parameters = {"Bucket": bucketName, "Prefix": "case_", "MaxKeys": 50000}
page_iterator = paginator.paginate(**operation_parameters)
allDownloadedCase = []
allCasedf = []
allPartDf = []
allFilesDf = []
isFirstCase = True
isFirstPart = True
isFirstFile = True

batchSize = 100
for page in tqdm(page_iterator, desc="page"):
    pageContent = page["Contents"]

    for i in tqdm(pageContent, desc="obj"):
        try:
            respRaw = cli.get_object(Bucket=bucketName, Key=i["Key"])

            caseDataJson = json.loads(respRaw["Body"].read())
            if caseDataJson["status"] != "OK":
                print(caseDataJson)
                continue
            caseId = i["Key"].split("_")[-1]

            caseMetadata = caseDataJson.get("data").get("data").get("main")
            partlistRaw = (
                caseDataJson.get("data").get("data").get("est_details").get("parts")
            )
            caseDf = pd.json_normalize([caseMetadata])
            partDf = pd.json_normalize(partlistRaw)
            partDf["CaseID"] = caseDf["CaseID"].item()
            allPartDf.append(partDf)
            allCasedf.append(caseDf)
            try:
                fileKey = f"files_{caseId}"
                respFileRaw = cli.get_object(Bucket=bucketName, Key=fileKey)
                filesJson = json.loads(respFileRaw["Body"].read())
                fileDf = pd.json_normalize(filesJson)
                fileDf["CaseID"] = caseDf["CaseID"].item()

                allFilesDf.append(fileDf)

            except Exception as e2:
                print(e2)

            if len(allPartDf) >= batchSize:
                rawPartDf = pd.concat(allPartDf)
                wr.s3.to_parquet(
                    df=rawPartDf,
                    path=f"s3://{partBucket}/",
                    dataset=True,
                    mode="overwrite" if isFirstPart else "append",
                    partition_cols=["CoType"],
                )
                allPartDf.clear()
                isFirstPart = False
                print("Written partlist to remote fs")
            if len(allCasedf) >= batchSize:
                print("Current ")
                rawCaseDf = pd.concat(allCasedf)
                wr.s3.to_parquet(
                    df=rawCaseDf,
                    path=f"s3://{caseBucket}/",
                    dataset=True,
                    mode="overwrite" if isFirstCase else "append",
                    partition_cols=["Vehicle_Type"],
                )
                isFirstCase = False

                allCasedf.clear()
                print("Written case to remote fs")
            if len(allFilesDf) >= batchSize:
                rawFilesDf = pd.concat(allFilesDf)
                wr.s3.to_parquet(
                    df=rawFilesDf,
                    path=f"s3://{fileBucket}/",
                    dataset=True,
                    mode="overwrite" if isFirstFile else "append",
                    partition_cols=["StdDocDesc"],
                )
                isFirstFile = False

                allFilesDf.clear()
                print("Written file df to remote fs")
        except Exception as e1:
            print(e1)
        # print(caseDataJson)
