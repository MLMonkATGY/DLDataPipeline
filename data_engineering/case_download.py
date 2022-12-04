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

bucketName = "mrm_raw"
cli = boto3.client("s3", **kwrgs)
# cli.create_bucket(Bucket=bucketName)
buckets = cli.list_buckets()
# uploadFile = (
#     r"C:\Users\alex\Desktop\infrastructure\download\download_tasks.csv"
# )
# cli.upload_file(uploadFile, bucketName, "download_tasks.csv")
# downloadUrl = cli.generate_presigned_url(
#     "get_object",
#     Params={"Bucket": bucketName, "Key": "download_tasks.csv"},
# )
def download_worker(caseIdList):
    kwrgs = {
        "endpoint_url": "http://192.168.1.4:8333",
        # "aws_access_key_id": accessKey,
        # "aws_secret_access_key": secretKey,
        # "Username": "aaa",
    }
    cli = boto3.client("s3", **kwrgs)
    successiveFailure = 0
    for caseId in caseIdList:
        try:
            if caseId in allDownloadedCase:
                continue
            url2 = f"http://10.1.1.50:4011/api/dsa/query/get_caseData?case_id={caseId}&get_ai_images=0&get_related_cases=0&get_est_details=1"
            r2 = requests.get(url2, allow_redirects=True, timeout=(1, 5))
            key2 = f"case_{caseId}"
            cli.put_object(Bucket=bucketName, Body=r2.content, Key=key2)
            url = f"http://10.1.1.50:3000/api/dsa/query/get_caseFiles?case_id={caseId}&get_file_info_only=1"
            r = requests.get(url, allow_redirects=True, timeout=(1, 5))
            key = f"files_{caseId}"
            cli.put_object(Bucket=bucketName, Body=r.content, Key=key)
            successiveFailure = 0
        except Exception as e1:
            print(e1)
            successiveFailure += 1
            if successiveFailure > 5:
                srcUrlIp = "10.1.1.50"
                pingTestResp = os.system("ping -c 1 " + srcUrlIp)
                if pingTestResp != 0:
                    raise Exception("Url IP not reachable")


# srcUrlIp = "10.1.1.50"
# pingTestResp = os.system("ping -c 1 -t1000" + srcUrlIp)
# if pingTestResp != 0:
#     raise Exception("Url IP not reachable")
if __name__ == "__main__":
    paginator = cli.get_paginator("list_objects")
    operation_parameters = {"Bucket": bucketName, "Prefix": "case_", "MaxKeys": 50000}
    page_iterator = paginator.paginate(**operation_parameters)
    allDownloadedCase = []
    for page in tqdm(page_iterator):
        pageContent = page["Contents"]
        downloadedCaseId = [int(x["Key"].split("_")[-1]) for x in pageContent]
        allDownloadedCase.extend(downloadedCaseId)
    startId = 9000000
    endId = 14000000
    allTargetCase = list(range(startId, endId))
    allCaseToDownload = list(set(allTargetCase).difference(set(allDownloadedCase)))
    downloadTaskbatch = []
    batchSize = 100
    workerNum = 10
    for i in range(0, len(allCaseToDownload), batchSize):
        downloadTaskbatch.append(allCaseToDownload[i : i + batchSize])
    Parallel(n_jobs=workerNum)(
        delayed(download_worker)(batchTask) for batchTask in tqdm(downloadTaskbatch)
    )
