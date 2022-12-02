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
paginator = cli.get_paginator("list_objects")
operation_parameters = {"Bucket": bucketName, "Prefix": "case_"}
page_iterator = paginator.paginate(**operation_parameters)
allDownloadedCase = []
for page in page_iterator:
    pageContent = page["Contents"]
    downloadedCaseId = [int(x["Key"].split("_")[-1]) for x in pageContent]
    allDownloadedCase.extend(downloadedCaseId)
startId = 10000000
endIdx = 16000000
for caseId in tqdm(range(startId, endIdx)):
    try:
        if caseId in allDownloadedCase:
            continue
        url = f"http://10.1.1.50:3000/api/dsa/query/get_caseFiles?case_id={caseId}&get_file_info_only=1"
        r = requests.get(url, allow_redirects=True)
        key = f"files_{caseId}"
        cli.put_object(Bucket=bucketName, Body=r.content, Key=key)
        url2 = f"http://10.1.1.50:4011/api/dsa/query/get_caseData?case_id={caseId}&get_ai_images=0&get_related_cases=0&get_est_details=1"
        r2 = requests.get(url2, allow_redirects=True)
        key2 = f"case_{caseId}"
        cli.put_object(Bucket=bucketName, Body=r2.content, Key=key2)

    except Exception as e1:
        print(e1)
