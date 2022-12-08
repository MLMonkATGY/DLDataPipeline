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

kwrgs = {
    "endpoint_url": "http://192.168.1.4:8333",
}
bucketName = "raw_imgs"
cli = boto3.client(
    "s3",
    **{
        "endpoint_url": "http://192.168.1.4:8333",
    }
)
paginator = cli.get_paginator("list_objects_v2")
operation_parameters = {"Bucket": bucketName}
page_iterator = paginator.paginate(**operation_parameters)
downloadedImgs = []
for page in tqdm(page_iterator):
    pageContent = page["Contents"]
    downloadedCaseId = set([int(x["Key"].split("_")[0]) for x in pageContent])
    downloadedImgs.extend(downloadedCaseId)
print(len(downloadedImgs))
