from dataclasses import dataclass
from typing import List


@dataclass(frozen=False)
class TrainClassiferParams:
    srcImgDir: str
    srcAnnFile: str
    imgSize: int
    trainBatchSize: int
    trainCPUWorker: int
    experimentName: str
    localSaveDir: str
    saveTopNBest: int
    check_val_every_n_epoch: int
    learningRate: float
    trainingPrecision: int
    randomSeed: int
    maxEpoch: int
    runName: str
    targetPart: str
    kFoldSplit: int
    version: int
    imgAngle: str
    not_dmg_label_count: int
    dmg_label_count: int
    posThreshold: float
    vehicleType: str
    tuningTimeout: int
    currentPosWeight: List[float]
    currentColName: List[str]


trainParams = TrainClassiferParams(
    vehicleType="Saloon - 4 Dr",
    srcImgDir="/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/imgs",
    srcAnnFile="",
    runName="wheel",
    targetPart="wheel",
    imgAngle="",
    version=1,
    imgSize=300,
    trainBatchSize=100,
    trainCPUWorker=10,
    experimentName="saloon_balance_rear",
    localSaveDir="mlruns",
    saveTopNBest=1,
    check_val_every_n_epoch=5,
    learningRate=1e-3,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=5,
    kFoldSplit=3,
    not_dmg_label_count=0,
    dmg_label_count=0,
    posThreshold=0.5,
    tuningTimeout=3600,
    currentPosWeight=[],
    currentColName=[],
)
