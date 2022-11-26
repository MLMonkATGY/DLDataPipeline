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
from GenLabels import GenLabels
from LocalPipeline import ExtractTrainTestData, getAllParts
from ViewData import readAndCombineDf, viewData
import ujson as json


@hydra.main(version_base=None, config_name="de_config")
def Remote_DE(cfg: ConfigParams) -> None:

    # ExtractTrainTestData(cfg)
    log.info("Start getting images")
    searchStr = f"{cfg.trainTestDataDir}/**/*.JPG"
    allImgs = glob.glob(searchStr, recursive=True)
    allLabelPath = []
    allPart = getAllParts(cfg)
    for viewName in cfg.targetDocDesc:
        labelPath = GenLabels(cfg, viewName, allImgs)
        allLabelPath.append(labelPath)

    labeldf = readAndCombineDf(allLabelPath)
    viewData(labeldf, allPart, cfg)


if __name__ == "__main__":
    Remote_DE()
