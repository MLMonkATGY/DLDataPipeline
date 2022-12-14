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

accessKey = "KIHI3SFQ0DNN7EAYM2E"
secretKey = "w/lrXU98nFEMI/J7MDENG/bP1RfiCYERA9GH"
# accessKey = "alextay96"
# secretKey = "Iamalextay96"

kwrgs = {
    "endpoint_url": "http://192.168.1.4:8333",
    "aws_access_key_id": accessKey,
    "aws_secret_access_key": secretKey,
    # "Username": "aaa",
}


def download_worker(caseIdList):
    s = requests.Session()

    kwrgs = {
        "endpoint_url": "http://192.168.1.4:8333",
        # "aws_access_key_id": accessKey,
        # "aws_secret_access_key": secretKey,
        # "Username": "aaa",
    }
    cli = boto3.client("s3", **kwrgs)

    successiveFailure = 0
    print(f"start : {caseIdList[0]}")
    for caseId in tqdm(caseIdList, desc="download"):
        try:
            url2 = f"http://10.1.1.50:4011/api/dsa/query/get_caseData?case_id={caseId}&get_ai_images=0&get_related_cases=0&get_est_details=1"
            r2 = s.get(url2, allow_redirects=True, timeout=(1, 5))
            key2 = f"case_{caseId}"
            cli.put_object(Bucket=bucketName, Body=r2.content, Key=key2)
            url = f"http://10.1.1.50:3000/api/dsa/query/get_caseFiles?case_id={caseId}&get_file_info_only=1"
            r = s.get(url, allow_redirects=True, timeout=(1, 5))
            key = f"files_{caseId}"
            cli.put_object(Bucket=bucketName, Body=r.content, Key=key)
            successiveFailure = 0
        except Exception as e1:
            print(e1)
            successiveFailure += 1
    print(f"end : {caseIdList[-1]}")


if __name__ == "__main__":
    bucketName = "mrm_raw"
    cli = boto3.client("s3", **kwrgs)
    # cli.create_bucket(Bucket=bucketName)
    buckets = cli.list_buckets()
    paginator = cli.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucketName, "Prefix": "case_"}
    page_iterator = paginator.paginate(**operation_parameters)
    allDownloadedCase = []
    for page in tqdm(page_iterator):
        pageContent = page["Contents"]
        downloadedCaseId = [int(x["Key"].split("_")[-1]) for x in pageContent]
        allDownloadedCase.extend(downloadedCaseId)
    startId = 10000000
    endId = 14000000
    allTargetCase = list(range(startId, endId))
    allCaseToDownload = list(set(allTargetCase).difference(set(allDownloadedCase)))
    downloadTaskbatch = []
    batchSize = 5000
    workerNum = 10
    for i in range(0, len(allCaseToDownload), batchSize):
        downloadTaskbatch.append(allCaseToDownload[i : i + batchSize])
    Parallel(n_jobs=workerNum)(
        delayed(download_worker)(batchTask)
        for batchTask in tqdm(downloadTaskbatch, desc="batch")
    )
