import pandas as pd
import pathlib
from Config import ConfigParams, log
import hydra
from hydra.core.config_store import ConfigStore
import logging
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import glob
from zipfile import ZipFile
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def genFilename(row):
    caseID = row["CaseID"]
    dociD = row["iDOCID"]
    return f"{caseID}_{dociD}.JPG"


def generate_df(cfg: ConfigParams):
    log.debug("Start process")
    baseOutputPath = pathlib.Path(cfg.writeOutputDir)
    caseDf = pd.read_parquet(cfg.caseDfPath)
    # print(caseDf.columns)
    caseCol = ["CaseID", "Vehicle_Type", "Model"]
    caseDf = caseDf[caseCol]
    fileDf = pd.read_parquet(cfg.fileDfpath)
    # print(fileDf["StdDocDesc"].value_counts())
    fileCol = ["CaseID", "iDOCID", "StdDocDesc", "FinalDate"]
    fileDf = fileDf[fileCol]
    imgFileDf = caseDf.merge(fileDf, on="CaseID")
    # print(imgFileDf)
    imgFileDf = imgFileDf[imgFileDf["StdDocDesc"].isin(cfg.targetDocDesc)]
    imgFileDf["FinalDate"] = pd.to_datetime(imgFileDf["FinalDate"])
    imgFileDf.sort_values(by="FinalDate", inplace=True)
    firstImgDf = imgFileDf.groupby(["CaseID", "StdDocDesc"]).head(1)
    firstImgDf["Filename"] = firstImgDf.apply(lambda x: genFilename(x), axis=1)
    # print(firstImgDf["Filename"])
    log.debug(caseDf["Vehicle_Type"].value_counts().head(10))

    validateDf = (
        firstImgDf.groupby("CaseID")["StdDocDesc"]
        .apply(list)
        .apply(lambda x: len(x) == len(set(x)))
    ).reset_index()
    assert len(validateDf[validateDf["StdDocDesc"] == False]) == 0
    outputFilePath = baseOutputPath / "valid_img_ds.parquet"
    firstImgDf.to_parquet(outputFilePath)
    log.info(f"Successfully write parquet : {outputFilePath}")
    return outputFilePath


def packageIntoDataset(imgDfPath, cfg: ConfigParams):
    baseOutputPath = pathlib.Path(cfg.writeOutputDir)
    table = pq.read_table(
        imgDfPath,
        columns=["CaseID", "Filename", "StdDocDesc", "Vehicle_Type"],
        # filters=[("StdDocDesc", "in", ["Front View"])],
    )
    dsRootPath = baseOutputPath / cfg.datasetName
    tempDf = table.to_pandas()
    print(tempDf["StdDocDesc"].value_counts())
    print(tempDf.columns)

    log.info(f"Created new df : {dsRootPath}")
    return dsRootPath


def ExtractWorker(cfg: ConfigParams, caseId: int, df: pd.DataFrame):
    info = df[df["CaseID"] == caseId]
    if info.empty:
        return
    srcZip = f"{cfg.imgSrcDir}/{caseId}.zip"
    allValidFiles = info["Filename"].tolist()
    try:
        with ZipFile(srcZip, "r") as zipObject:
            listOfFileNames = zipObject.namelist()
            for fileName in listOfFileNames:
                if fileName in allValidFiles:
                    zipObject.extract(fileName, cfg.imgSinkDir)
    except Exception as e1:
        print(e1)


def ExtractImgFromZip(dfPath: str, cfg: ConfigParams):
    srcImgDir = pathlib.Path(cfg.imgSrcDir)
    allZipFile = glob.glob(cfg.imgSrcDir + "**/*.zip", recursive=True)
    allCaseId = [int(x.split("/")[-1].split(".")[0]) for x in allZipFile]
    table = pq.read_table(
        dfPath,
        columns=["CaseID", "Filename", "StdDocDesc", "Vehicle_Type"],
        filters=[
            ("CaseID", "in", allCaseId),
            ("Vehicle_Type", "=", cfg.targetVehicleType),
        ],
    )

    df: pd.DataFrame = table.to_pandas()
    df.drop_duplicates(subset="Filename", inplace=True)
    allCaseIdToExtract = df["CaseID"].unique().tolist()
    Parallel(n_jobs=cfg.extractionWorker)(
        delayed(ExtractWorker)(cfg=cfg, caseId=caseId, df=df)
        for caseId in allCaseIdToExtract
    )


def CompressDataset(cfg: ConfigParams):

    imgSrcDir = cfg.imgSinkDir
    outputDir = pathlib.Path(cfg.trainTestDataDir)
    zip_name = outputDir / f"{cfg.trainTestDataFilename}"

    with ZipFile(zip_name, "w") as zip_ref:
        for folder_name, subfolders, filenames in tqdm(
            os.walk(imgSrcDir), desc="zipping files"
        ):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, imgSrcDir))

    zip_ref.close()


def ExtractTrainTestData(cfg: ConfigParams):

    outputDir = pathlib.Path(cfg.trainTestDataDir)
    zip_name = outputDir / f"{cfg.trainTestDataFilename}"
    os.makedirs(outputDir / "train_test_img", exist_ok=True)
    with ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(outputDir / "train_test_img")


@hydra.main(version_base=None, config_name="de_config")
def my_app(cfg: ConfigParams) -> None:
    imgDfPath = generate_df(cfg)
    # imgDfPath = (
    #     "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/valid_img_ds.parquet"
    # )
    ExtractImgFromZip(imgDfPath, cfg)
    CompressDataset(cfg)
    ExtractTrainTestData(cfg)


if __name__ == "__main__":
    my_app()
