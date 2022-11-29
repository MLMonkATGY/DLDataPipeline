from typing import List
import pandas as pd
import plotly.express as px
import shutil
import glob
import copy
from tqdm import tqdm
import ujson as json
import pandas as pd
import pathlib
from Config import ConfigParams, log
import hydra
import logging
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import glob
from zipfile import ZipFile
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def GenLabels(cfg: ConfigParams, viewAngle: str, allImgs: List[str]):
    partlistDf = pd.read_parquet(cfg.rawPartlistGroup)
    validImgDf = pd.read_parquet(cfg.validImgDsPath)
    print(validImgDf.columns)
    vTypeName = cfg.targetVehicleType.replace(" ", "")
    baseOutputDir = pathlib.Path(cfg.writeOutputDir)

    allLocalCaseId = set([int(x.split("/")[-1].split("_")[0]) for x in allImgs])
    partlistDf = partlistDf[partlistDf["CaseID"].isin(allLocalCaseId)]
    labelDir = baseOutputDir / cfg.outputLabelDir
    os.makedirs(labelDir, exist_ok=True)

    with open(cfg.imgAngleTopartMap, "r") as f:
        viewToPart = json.load(f)

    targetPart = viewToPart[viewAngle]
    allValidImgInAngle = validImgDf[validImgDf["StdDocDesc"] == viewAngle][
        "Filename"
    ].tolist()
    templatePayload = {x: 0 for x in targetPart}
    # allImgsInLocal = glob.glob(searchStr, recursive=True)

    localCaseImgMap = dict()
    for filename in tqdm(allValidImgInAngle):
        path = f"{cfg.trainTestDataDir }/{filename}"
        caseId = int(filename.split("_")[0])
        localCaseImgMap[caseId] = {
            "CaseID": caseId,
            "ViewAngle": viewAngle,
            "Filename": filename,
            "Path": path,
        }
    allPayload = []
    allGeneratedLabelPath = []
    for caseId in tqdm(partlistDf["CaseID"].unique()):
        if caseId not in localCaseImgMap.keys():
            continue
        dmgParts = (
            partlistDf[partlistDf["CaseID"] == caseId]["lvl_3_desc"].unique().tolist()
        )
        template = copy.deepcopy(templatePayload)
        template["CaseID"] = caseId
        template["ViewAngle"] = localCaseImgMap[caseId]["ViewAngle"]
        template["Filename"] = localCaseImgMap[caseId]["Filename"]
        template["Path"] = localCaseImgMap[caseId]["Path"]

        for target in targetPart:
            if target in dmgParts:
                template[target] = 1
        allPayload.append(template)
    imgLabelDf = pd.json_normalize(allPayload)
    outputLabelFilename = f"{labelDir}/{vTypeName}_{viewAngle}_img_label.csv"
    imgLabelDf.to_csv(outputLabelFilename)
    log.info(f"Successfully created label : {outputLabelFilename}")
    return outputLabelFilename
