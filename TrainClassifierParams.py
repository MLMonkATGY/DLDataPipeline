from dataclasses import dataclass


@dataclass(frozen=False)
class TrainClassiferParams:
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
    expId: int


trainParams = TrainClassiferParams(
    srcAnnFile="",
    runName="wheel",
    targetPart="wheel",
    imgAngle="",
    version=1,
    imgSize=480,
    trainBatchSize=40,
    trainCPUWorker=8,
    experimentName="dmg_multilabel",
    expId=66,
    localSaveDir="mlruns",
    saveTopNBest=3,
    check_val_every_n_epoch=1,
    learningRate=1e-3,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=10,
    kFoldSplit=5,
    not_dmg_label_count=0,
    dmg_label_count=0,
)
