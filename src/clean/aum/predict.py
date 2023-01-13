import pandas as pd
import awswrangler as wr
from pathlib import Path
import glob
import ujson as json
from joblib import Parallel, delayed
from tqdm import tqdm
import requests
import os
import yaml
from pprint import pprint
from src.clean.aum.dataset import PredictionImageDataset

from src.clean.aum.train import ProcessModel, create_model
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader2
from loguru import logger
from torchmetrics import Accuracy, ConfusionMatrix

# from torchdata.datapipes.iter import IterableWrapper

# dp = IterableWrapper(["s3://BUCKET_NAME"]).list_files_by_fsspec()
wr.config.s3_endpoint_url = "http://192.168.1.4:8333"


def transport_worker(df: pd.DataFrame, sinkDir: str):
    s = requests.Session()
    for _, i in tqdm(df.iterrows(), desc="files"):
        url = i["url"]
        imgBytes = s.get(url)
        localPath = os.path.join(sinkDir, i["filename"])
        with open(localPath, "wb") as f:
            f.write(imgBytes.content)


def get_src_file():
    srcDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/aum_high_resolution"
    downloadedCsv = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/downloaded_case_id.json"

    search = f"{srcDir}/**/pred_cleaned_*.csv"
    allSrcData = glob.glob(search, recursive=True)
    allSrcData = sorted(allSrcData)
    allDf = []
    for i in allSrcData:
        allDf.append(pd.read_csv(i))
    allCleanDf = pd.concat(allDf)
    allCleanDf["CaseID"] = allCleanDf.apply(
        lambda x: int(x["filename"].split("_")[0]), axis=1
    )
    # allCleanDf = allCleanDf[allCleanDf["dataset_index"] < 2]
    vehicleType = ["Saloon - 4 Dr", "Hatchback - 5 Dr", "SUV - 5 Dr"]

    def myFilter(x):
        return x["Vehicle_Type"] in vehicleType

    caseDf = wr.s3.read_parquet(
        f"s3://scope_case/",
        partition_filter=myFilter,
        dataset=True,
        columns=["CaseID", "Vehicle_Type", "Model"],
    )
    allCleanDf = allCleanDf.merge(caseDf, on="CaseID")
    print(len(allCleanDf["CaseID"].unique()))
    uniqueCaseDf = allCleanDf.drop_duplicates(subset=["CaseID"])
    print(uniqueCaseDf["Vehicle_Type"].value_counts().reset_index())
    print(
        uniqueCaseDf[["Vehicle_Type", "Model"]]
        .value_counts(ascending=False)
        .reset_index()
        .head(30)
    )
    labelDf = wr.s3.read_parquet(path=f"s3://multilabel_df/", dataset=True)
    allCleanDf = allCleanDf.merge(labelDf, on="CaseID")
    # print(allCleanDf)
    filesDf = wr.s3.read_parquet(
        f"s3://scope_file/", dataset=True, columns=["CaseID", "iDOCID"]
    )
    filesDf = filesDf[filesDf["CaseID"].isin(allCleanDf["CaseID"].unique().tolist())]
    filesDf["filename"] = filesDf[["CaseID", "iDOCID"]].apply(
        lambda x: str(int(x["CaseID"])) + "_" + str(int(x["iDOCID"])) + ".JPG",
        axis=1,
    )
    filesDf.to_parquet("./local_files_metadata.parquet")


def download_files():
    filesDf = pd.read_parquet(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/local/local_files_metadata.parquet"
    )
    filesDf["filename"] = filesDf[["CaseID", "iDOCID"]].apply(
        lambda x: str(int(x["CaseID"])) + "_" + str(int(x["iDOCID"])) + ".JPG",
        axis=1,
    )
    print(filesDf)
    endpoint = "http://192.168.1.4:8888/buckets/raw_imgs/"

    filesDf.sort_values(by="StdDocDesc", inplace=True)
    filesDf["url"] = endpoint + filesDf["filename"]
    localImgSink = (
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"
    )
    print("Start scanning local files")

    localImgs = glob.glob(f"{localImgSink}/**.JPG", recursive=True)
    allLocalFilename = [x.split("/")[-1] for x in localImgs]
    filesDf = filesDf[~filesDf["filename"].isin(allLocalFilename)]
    print(len(filesDf))
    batchSize = 1000
    batchTask = []

    for i in range(0, len(filesDf), batchSize):
        batchTask.append(filesDf.iloc[i : i + batchSize])
    Parallel(n_jobs=5)(
        delayed(transport_worker)(taskDf, localImgSink)
        for taskDf in tqdm(batchTask, desc="tasks")
    )


def load_models():
    search = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/aum_v2_logs/lightning_logs"
    allHParams = glob.glob(f"{search}/**/*.yaml", recursive=True)
    allModel = []
    for hparamsFile in tqdm(allHParams):
        with open(hparamsFile, "r") as f:
            content = yaml.load(f, Loader=yaml.Loader)
            if content["lr"] == 1e-3:
                modelName = content["part"]
                modelDir = "/".join(hparamsFile.split("/")[:-1]) + "/checkpoints"
                # print(modelDir)
                modelPath = glob.glob(f"{modelDir}/**.ckpt")
                allModel.append({"model_path": modelPath, "model_name": modelName})
                # print(allModel)
    with open("./model_dataset_aum_v2.json", "w") as f:
        json.dump(allModel, f)


def predict_part(model, dataloader, model_name):
    modelOutDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/aum_v2_model_script"
    model.eval()
    device = torch.device("cuda")
    # model = torch.jit.trace(model, torch.rand((120, 3, 480, 480)))
    model = model.half()
    model = model.to(device)

    model = model.to_torchscript(
        method="trace",
        example_inputs=torch.rand((200, 3, 480, 480), dtype=torch.float16).to(device),
        file_path=f"{modelOutDir}/{model_name}",
    )
    model = torch.jit.freeze(model)
    model = torch.jit.optimize_for_inference(model)
    allPredInfo = []
    part = dataloader.dataset.targetName
    view = dataloader.dataset.targetView
    accMetrics = Accuracy(task="multiclass", num_classes=2).to(device)
    confMatMetric = ConfusionMatrix(
        task="multiclass", num_classes=2, normalize="true"
    ).to(device)
    with torch.no_grad():
        for batchId, batch in enumerate(tqdm(dataloader, desc="predict")):
            imgs = batch["img"].to(device)
            # print(imgs.shape)
            files = batch["filename"]
            vehicleType = batch["vehicleType"]
            vehicleModel = batch["model"]
            with autocast():
                logit = model(imgs)
            preds = torch.argmax(logit, dim=1)
            labels = batch["target"].type(torch.uint8).to(device)
            accMetrics.update(preds, labels)
            confMatMetric.update(preds, labels)
            predProbs = torch.softmax(logit, dim=1)
            for p, f, gt, probs, vType, vModel in zip(
                preds, files, labels, predProbs, vehicleType, vehicleModel
            ):
                softmaxProbs = probs.cpu().numpy().tolist()
                info = {
                    "pred": p.tolist(),
                    "gt": gt.tolist(),
                    "probs_0": softmaxProbs[0],
                    "probs_1": softmaxProbs[1],
                    "pred_probs": softmaxProbs[p],
                    "filename": f,
                    "part": part,
                    "view": view,
                    "vehicleType": vType,
                    "model": vModel,
                }
                allPredInfo.append(info)
            if batchId % 100 == 0:
                confMat = confMatMetric.compute()
                logger.success(f"TNR : {confMat[0][0]}")
                logger.success(f"TPR : {confMat[1][1]}")
                logger.success(f"ACC : {accMetrics.compute()}")

        predDf = pd.json_normalize(allPredInfo)
    return predDf


def predict_all():
    with open("./model_dataset_aum_v2.json", "r") as f:
        allModelAnn = json.load(f)
    filesDf = pd.read_parquet(
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/local/local_files_metadata.parquet"
    )

    evalTransform = A.Compose(
        [
            A.LongestMaxSize(480),
            A.PadIfNeeded(
                min_height=480,
                min_width=480,
                border_mode=0,
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )
    labelDf = wr.s3.read_parquet(path=f"s3://multilabel_df_lvl_3/", dataset=True)
    print(labelDf.columns)
    img_dir = (
        "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"
    )
    outputDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/predict_aum_v2"
    os.makedirs(outputDir, exist_ok=True)
    for i in tqdm(allModelAnn, desc="part_view"):
        model = create_model(True)

        # model = ProcessModel(model, "", 1e-3, False)
        model = ProcessModel.load_from_checkpoint(
            # part, lr, isFiltered
            checkpoint_path=i["model_path"][0],
            model=model,
            part="",
            lr=1e-3,
            isFiltered=False,
        )
        targetView = i["model_name"].replace("_img_labels", "").split("_")[-1]
        targetPart = (
            i["model_name"].replace("_img_labels", "").replace(targetView, "")[:-1]
        )

        targetFilesDf = filesDf[filesDf["StdDocDesc"] == targetView]
        partLabelDf = labelDf[
            labelDf["CaseID"].isin(filesDf["CaseID"].unique().tolist())
        ][[targetPart, "CaseID", "Vehicle_Type", "Model"]]
        targetFilesDf: pd.DataFrame = targetFilesDf.merge(partLabelDf, on="CaseID")
        targetFilesDf.sort_values(by="CaseID", ascending=True, inplace=True)
        print(targetFilesDf.columns)
        # targetFilesDf = targetFilesDf.groupby(
        #     ["Vehicle_Type", "Model", targetPart]
        # ).head(20)
        # modelTypeLabelDf = (
        #     targetFilesDf.groupby(["Model"])["CaseID"]
        #     .count()
        #     .reset_index()
        #     .rename(columns={"CaseID": "count"})
        #     .sort_values(by="count", ascending=False)
        #     .head(200)
        # )
        # targetFilesDf = (
        #     targetFilesDf[
        #         targetFilesDf["Model"].isin(modelTypeLabelDf["Model"].unique().tolist())
        #     ]
        #     .sort_values(by="CaseID", ascending=True)
        #     .groupby([targetPart, "Model"])
        #     .head(100)
        # )
        print(targetFilesDf.groupby(["Vehicle_Type", targetPart]).count().reset_index())
        # balanceTempDf = (
        #     targetFilesDf.groupby(["Vehicle_Type", "Model", targetPart])
        #     .count()
        #     .reset_index()
        #     .rename(columns={"CaseID": "count"})
        #     .sort_values(by="count", ascending=False)
        # )
        # print(balanceTempDf.tail(30))
        logger.success(f"Pred Set Size {len(targetFilesDf)}")
        ds = PredictionImageDataset(
            targetFilesDf, img_dir, targetPart, targetView, evalTransform
        )
        evalLoader = DataLoader2(
            ds,
            shuffle=False,
            batch_size=200,
            num_workers=12,
        )
        predDf = predict_part(model, evalLoader, i["model_name"])
        predOutputFile = f"{targetPart}_{targetView}.parquet"
        predDf.to_parquet(f"{outputDir}/{predOutputFile}")


def build_multi_view_multilabel_df():
    srcDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/predict_aum_v2"
    predLabelOutDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/pred_multilabel_aum_v2"
    os.makedirs(predLabelOutDir, exist_ok=True)
    allPredFiles = glob.glob(f"{srcDir}/*.parquet", recursive=True)
    allDf = []
    for i in allPredFiles:
        allDf.append(pd.read_parquet(i))
    completePredDf = pd.concat(allDf)
    completePredDf["CaseID"] = completePredDf["filename"].apply(
        lambda x: int(x.split("_")[0])
    )
    allViews = completePredDf["view"].unique().tolist()
    print(completePredDf[["view", "vehicleType", "CaseID"]])
    # allParts = completePredDf["part"].unique().tolist()
    partViewDf = completePredDf.groupby(["view", "vehicleType"]).head(1).reset_index()
    print(partViewDf)
    for _, row in partViewDf.iterrows():
        # part = row["part"]
        view = row["view"]
        vehicleType = row["vehicleType"]
        viewVehicleTypePredDf = completePredDf[
            (completePredDf["view"] == view)
            # & (completePredDf["part"] == part)
            & (completePredDf["vehicleType"] == vehicleType)
        ]

        relatedParts = sorted(viewVehicleTypePredDf["part"].unique().tolist())
        print(viewVehicleTypePredDf.columns)
        allOutputDfByPart = []
        for part in relatedParts:
            outputDf = pd.DataFrame()

            partPredDf = viewVehicleTypePredDf[viewVehicleTypePredDf["part"] == part]
            matchDf = partPredDf[partPredDf["pred"] == partPredDf["gt"]]
            acc = len(matchDf) / len(partPredDf)
            print(f"Acc : {acc}  Part : {part} VType : {vehicleType} : View : {view}")
            outputLabelName = f"vision_{part}"
            outputDf[outputLabelName] = partPredDf["pred"].values
            outputDf["CaseID"] = partPredDf["CaseID"].values
            outputDf["filename"] = partPredDf["filename"].values
            outputDf["view"] = partPredDf["view"].values
            outputDf["model"] = partPredDf["model"].values

            allOutputDfByPart.append(outputDf)
        multilabelDf: pd.DataFrame = allOutputDfByPart[0]
        for df in allOutputDfByPart[1:]:
            multilabelDf = multilabelDf.merge(
                df, on=["filename", "CaseID", "view", "model"]
            )

        print(multilabelDf.columns)
        outCsv = f"{predLabelOutDir}/{vehicleType}_{view.lower()}_img_labels.csv"
        multilabelDf.to_csv(outCsv)


def sample_label_df():
    srcDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/pred_multilabel_aum_v2"
    outDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/balance_multilabel_aum_v2"
    os.makedirs(outDir, exist_ok=True)
    allPredLabelCsv = glob.glob(f"{srcDir}/**.csv", recursive=True)
    for dfCsv in allPredLabelCsv:
        filename = dfCsv.split("/")[-1]

        rawLabelDf = pd.read_csv(dfCsv)
        balanceLabelDf = rawLabelDf.copy(deep=True)
        posLabelForView = {}
        partToOverSample = []
        for targetCol in rawLabelDf.filter(regex="vision_*").columns:
            labelPosDistrib = rawLabelDf[rawLabelDf[targetCol] == 1]
            posLabelsRatio = len(labelPosDistrib) / len(rawLabelDf)
            posLabelForView[targetCol] = posLabelsRatio
            if posLabelsRatio < 0.2:
                partToOverSample.append(targetCol)
        if len(partToOverSample) > 0:
            logger.success(f"Part : {filename}")
            pprint(posLabelForView)

        for part in partToOverSample:
            labelPosDistrib = rawLabelDf[rawLabelDf[part] == 1]
            balanceLabelDf = pd.concat([balanceLabelDf, labelPosDistrib])
            balanceLabelDf = pd.concat([balanceLabelDf, labelPosDistrib])
            labelPosDistrib = balanceLabelDf[balanceLabelDf[part] == 1]
            posLabelsRatio = len(labelPosDistrib) / len(balanceLabelDf)
            posLabelForView[part] = posLabelsRatio
        if len(partToOverSample) > 0:
            pprint(posLabelForView)
        outCsv = os.path.join(outDir, filename)
        balanceLabelDf.to_csv(outCsv)
        # print()


# get_src_file()
# download_files()

load_models()
predict_all()
build_multi_view_multilabel_df()
sample_label_df()


# def build_perfect_dataset(partName):
#     pass
