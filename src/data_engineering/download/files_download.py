import requests
import pandas as pd
import os
from tqdm import tqdm
import glob
import ujson as json
import pandas as pd
import boto3
import io
import ujson as json
from joblib import Parallel, delayed
from boto3.session import Session
from loguru import logger
import awswrangler as wr


def download_worker(caseIdList, outputDir):
    s = requests.Session()
    successiveFailure = 0
    print(f"start : {caseIdList[0]}")
    for caseId in tqdm(caseIdList, desc="download"):
        try:
            url = f"http://10.1.1.50:4011/api/dsa/query/get_caseFiles?case_id={caseId}"
            r = s.get(url, allow_redirects=True)
            with open(f"{outputDir}/{caseId}.zip", "wb") as f:
                f.write(r.content)
        except Exception as e1:
            print(e1)
            successiveFailure += 1
    print(f"end : {caseIdList[-1]}")


if __name__ == "__main__":
    kwrgs = {
        "endpoint_url": "http://localhost:8333",
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
    }

    cli = boto3.client("s3", **kwrgs)
    # cli.create_bucket(Bucket=bucketName)
    outputDir = r"D:\tmp_download"
    os.makedirs(outputDir, exist_ok=True)
    localFiles = os.listdir(outputDir)
    localFilesCaseId = [int(x.split(".")[0]) for x in localFiles]
    wr.config.s3_endpoint_url = "http://localhost:8333"
    bucket2 = "scope_case"
    caseDf = wr.s3.read_parquet(
        f"s3://{bucket2}/", dataset=True, columns=["CaseID", "Vehicle_Type"]
    )
    targetVehicleType = "Saloon - 4 Dr"
    caseDf = caseDf[caseDf["Vehicle_Type"] == targetVehicleType]

    validCaseToDownload = caseDf["CaseID"].unique().tolist()

    downloadTaskbatch = []
    batchSize = 20
    workerNum = 5
    downloadedImgs = []
    paginator = cli.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": "raw_imgs"}
    page_iterator = paginator.paginate(**operation_parameters)
    for page in tqdm(page_iterator):
        pageContent = page["Contents"]
        downloadedCaseId = set([int(x["Key"].split("_")[0]) for x in pageContent])
        downloadedImgs.extend(downloadedCaseId)
    downloadedImgs = sorted(downloadedImgs)
    allCaseToDownload = set(validCaseToDownload).difference(set(downloadedImgs))
    allCaseToDownload = allCaseToDownload.difference(set(localFilesCaseId))
    allCaseToDownload = sorted(allCaseToDownload)

    for i in range(0, len(allCaseToDownload), batchSize):
        downloadTaskbatch.append(allCaseToDownload[i : i + batchSize])
    Parallel(n_jobs=workerNum)(
        delayed(download_worker)(batchTask, outputDir)
        for batchTask in tqdm(downloadTaskbatch, desc="batch")
    )
