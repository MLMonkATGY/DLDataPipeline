import pandas as pd
import pathlib
from Config import ConfigParams, log
import hydra
from hydra.core.config_store import ConfigStore
import logging
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def generate_df(cfg: ConfigParams):
    log.debug("Start process")
    baseInputPath = pathlib.Path(cfg.srcDataInputDir)
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
    targetDocDesc = [
        "Front View",
        "Rear View",
        "Front View Left",
        "Front View Right",
        "Rear View Left",
        "Rear View Right",
    ]
    imgFileDf = imgFileDf[imgFileDf["StdDocDesc"].isin(targetDocDesc)]
    imgFileDf["FinalDate"] = pd.to_datetime(imgFileDf["FinalDate"])
    imgFileDf.sort_values(by="FinalDate", inplace=True)
    firstImgDf = imgFileDf.groupby(["CaseID", "StdDocDesc"]).head(1)
    validateDf = (
        firstImgDf.groupby("CaseID")["StdDocDesc"]
        .apply(list)
        .apply(lambda x: len(x) == len(set(x)))
    ).reset_index()
    assert len(validateDf[validateDf["StdDocDesc"] == False]) == 0
    outputFilePath = baseOutputPath / "valid_img.parquet"
    firstImgDf.to_parquet(outputFilePath)
    log.info(f"Successfully write parquet : {outputFilePath}")
    return outputFilePath


def packageIntoDataset(imgDfPath, cfg: ConfigParams):
    baseOutputPath = pathlib.Path(cfg.writeOutputDir)
    table = pq.read_table(imgDfPath)
    dsRootPath = baseOutputPath / cfg.datasetName

    pq.write_to_dataset(
        table,
        dsRootPath,
        partition_cols=["Vehicle_Type", "StdDocDesc"],
    )
    readDs = ds.dataset(dsRootPath, format="parquet")
    log.info(f"Created new df : {dsRootPath}")
    return dsRootPath


@hydra.main(version_base=None, config_name="de_config")
def my_app(cfg: ConfigParams) -> None:
    # pork should be port!
    imgDfPath = generate_df(cfg)
    dsRootPath = packageIntoDataset(imgDfPath, cfg)


if __name__ == "__main__":
    my_app()
