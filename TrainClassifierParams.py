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
    imgSize=300,
    trainBatchSize=80,
    trainCPUWorker=6,
    experimentName="test_remote_access",
    expId=64,
    localSaveDir="mlruns",
    saveTopNBest=3,
    check_val_every_n_epoch=1,
    learningRate=1e-2,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=5,
    kFoldSplit=3,
    not_dmg_label_count=0,
    dmg_label_count=0,
)