import pycurl
import io
from weed.filer import WeedFiler

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
import boto3

bucketName = "b1"
cli = boto3.client("s3", **kwrgs)
# cli.create_bucket(Bucket=bucketName)
buckets = cli.list_buckets()
uploadFile = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data_engineering/seaweedfs-compose.yml"
cli.upload_file(uploadFile, bucketName, "seaweedfs-compose.yml")
# cli.download_file(bucketName, "seaweedfs-compose.yml", "downloaded.yml")
downloadUrl = cli.generate_presigned_url(
    "get_object",
    Params={"Bucket": bucketName, "Key": "seaweedfs-compose.yml"},
)
print(downloadUrl)
