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

    kwrgs = {
        "endpoint_url": "http://localhost:8333",
    }
    cli = boto3.client("s3", **kwrgs)
    successiveFailure = 0
    print(f"start : {caseIdList[0]}")
    for caseId in tqdm(caseIdList, desc="download"):
        try:
            paginator = cli.get_paginator("list_objects_v2")
            operation_parameters = {"Bucket": bucketName, "Prefix": f"{caseId}_"}
            page_iterator = paginator.paginate(**operation_parameters)
            for page in page_iterator:
                if page["KeyCount"] > 0:
                    raise Exception(f"Case file alreadu exist : {caseId}")
                else:
                    break
            url = f"http://10.1.1.50:4011/api/dsa/query/get_caseFiles?case_id={caseId}"
            r = requests.get(url, allow_redirects=True)
            with open(f"{outputDir}/{caseId}.zip", "wb") as f:
                f.write(r.content)
        except Exception as e1:
            print(e1)
            successiveFailure += 1
    print(f"end : {caseIdList[-1]}")


if __name__ == "__main__":
    bucketName = "raw_imgs"
    cli = boto3.client("s3", **kwrgs)
    # cli.create_bucket(Bucket=bucketName)
    outputDir = r""
    os.makedirs(outputDir, exist_ok=True)
    endId = 14000000
    wr.config.s3_endpoint_url = "http://192.168.1.4:8333"
    bucket2 = "scope_case"
    caseDf = wr.s3.read_parquet(f"s3://{bucket2}/", columns=["CaseID"])
    allCaseToDownload = caseDf["CaseID"].unique().tolist()
    downloadTaskbatch = []
    batchSize = 5000
    workerNum = 1
    for i in range(0, len(allCaseToDownload), batchSize):
        downloadTaskbatch.append(allCaseToDownload[i : i + batchSize])
    Parallel(n_jobs=workerNum)(
        delayed(download_worker)(batchTask, outputDir)
        for batchTask in tqdm(downloadTaskbatch, desc="batch")
    )
