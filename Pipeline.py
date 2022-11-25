import pandas as pd
import pathlib
from Config import ConfigParams, log
import hydra
from hydra.core.config_store import ConfigStore
import logging
import pyarrow.dataset as ds
import pyarrow.parquet as pq


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
    print(firstImgDf["Filename"])
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


@hydra.main(version_base=None, config_name="de_config")
def my_app(cfg: ConfigParams) -> None:
    imgDfPath = generate_df(cfg)
    # imgDfPath = generate_df(cfg)

    # imgDfPath = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/valid_img_ds_partition"
    # dsRootPath = packageIntoDataset(imgDfPath, cfg)


if __name__ == "__main__":
    my_app()
