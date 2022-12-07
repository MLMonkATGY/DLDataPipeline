from dataclasses import dataclass, field
from typing import List
from hydra.core.config_store import ConfigStore
import logging


@dataclass
class ConfigParams:
    srcDataInputDir: str = (
        "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=d$/batch_download"
    )
    writeOutputDir: str = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data"
    caseDfPath: str = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/raw_src/complete_encoded_no_limit.parquet"
    fileDfpath: str = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/raw_src/file_metadata.parquet"
    validImgDsPath: str = (
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/valid_img_ds.parquet"
    )
    datasetName: str = "dmg_vision_dataset"
    imgSrcDir: str = (
        "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=d$/batch_download"
    )
    imgSinkDir: str = "/home/alextay96/Deep_Learning_Data/mrm_imgs"
    trainTestDataDir: str = (
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_img"
    )
    compressImgPath: str = (
        "/home/alextay96/Deep_Learning_Data/mrm_imgs_remote_sink/train_test_data.zip"
    )
    trainTestDataFilename: str = "train_test_data.zip"

    targetVehicleType: str = "Saloon - 4 Dr"
    targetDocDesc: list[str] = field(
        default_factory=lambda: [
            "Front View",
            "Rear View",
            "Front View Left",
            "Front View Right",
            "Rear View Left",
            "Rear View Right",
        ]
    )
    extractionWorker: int = 8
    rawPartlistGroup: str = "/home/alextay96/Desktop/new_workspace/partlist_prediction/data/processed/best_2/partlist_3lvl_2.parquet"
    imgAngleTopartMap: str = "/home/alextay96/Desktop/new_workspace/partlist_prediction/data/processed/best_2/angle.json"
    outputLabelDir: str = "train_test_labels"


cs = ConfigStore.instance()
cs.store(name="de_config", node=ConfigParams)

log = logging.getLogger(__name__)
log.setLevel("DEBUG")
