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
    vehicleType="SUV - 5 Dr",
    srcImgDir="/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs",
    srcAnnFile="",
    runName="wheel",
    targetPart="wheel",
    imgAngle="",
    version=1,
    imgSize=480,
    trainBatchSize=24,
    trainCPUWorker=10,
    experimentName="pred_labels_v5",
    localSaveDir="mlruns",
    saveTopNBest=1,
    check_val_every_n_epoch=1,
    learningRate=5e-4,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=30,
    kFoldSplit=2,
    not_dmg_label_count=0,
    dmg_label_count=0,
    posThreshold=0.5,
    tuningTimeout=3600,
    currentPosWeight=[],
    currentColName=[],
)
