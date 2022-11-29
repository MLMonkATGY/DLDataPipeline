from dataclasses import dataclass


@dataclass(frozen=False)
class TrainClassiferParams:
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
    runName="wheel",
    targetPart="wheel",
    imgAngle="",
    version=1,
    imgSize=640,
    trainBatchSize=64,
    trainCPUWorker=6,
    experimentName="clean_dataset",
    expId=65,
    localSaveDir="mlruns",
    saveTopNBest=2,
    check_val_every_n_epoch=2,
    learningRate=1e-3,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=10,
    kFoldSplit=10,
    not_dmg_label_count=0,
    dmg_label_count=0,
)
