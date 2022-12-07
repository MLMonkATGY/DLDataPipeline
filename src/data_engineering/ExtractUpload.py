from joblib import Parallel, delayed
import io
from tqdm import tqdm
import glob
from zipfile import ZipFile
from pprint import pprint
import boto3
import functools
import pyarrow as pa
import os
from typing import List
import awswrangler as wr
import pandas as pd

pd.set_option("display.max_rows", 300)


def worker(allLocalZip: List, fileDf: pd.DataFrame):
    kwrgs = {
        "endpoint_url": "http://localhost:8333",
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        # "Username": "aaa",
    }
    rawImgBucket = "raw_imgs"

    cli = boto3.client("s3", **kwrgs)
    for zipFile in tqdm(allLocalZip, desc="files"):
        try:
            localCaseId = int(zipFile.split(os.sep)[-1].split(".")[0])
            targetFilesRow = fileDf[fileDf["CaseID"] == localCaseId]

            if targetFilesRow.empty:
                continue

            targetFilesInCase = targetFilesRow["iDOCID"].tolist()
            targetFilename = [
                f"{localCaseId}_{int(docid)}.JPG" for docid in targetFilesInCase
            ]
            completeUpload = 0
            with ZipFile(zipFile, "r") as zip:
                for i in targetFilename:
                    with zip.open(i, "r") as file_data:
                        bytes_content = file_data.read()
                        file_like_object = io.BytesIO(bytes_content)
                        cli.put_object(
                            Bucket=rawImgBucket, Key=i, Body=file_like_object
                        )
                        completeUpload += 1
            if completeUpload != len(targetFilename):
                print("Upload images not complete")
        except Exception as e1:
            print(e1)


if __name__ == "__main__":
    kwrgs = {
        "endpoint_url": "http://localhost:8333",
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        # "Username": "aaa",
    }
    rawImgBucket = "raw_imgs"

    cli = boto3.client("s3", **kwrgs)
    # cli.create_bucket(Bucket=rawImgBucket)

    bucket = "scope_file"

    wr.config.s3_endpoint_url = "http://localhost:8333"

    fileDf = wr.s3.read_parquet(
        f"s3://{bucket}/",
        dataset=True,
    )
    fileDf.sort_values(by="CaseID", inplace=True)
    srcDir = r"D:\batch_download"
    allLocalZip = glob.glob(f"{srcDir}/**/*.zip", recursive=True)
    allTaskBatch = []
    batchSize = 30
    for i in range(0, len(allLocalZip), batchSize):
        allTaskBatch.append(allLocalZip[i: i + batchSize])
    Parallel(n_jobs=10)(delayed(worker)(tasks, fileDf)
                        for tasks in tqdm(allTaskBatch, desc="batch"))
