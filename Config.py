from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import logging


@dataclass
class ConfigParams:
    srcDataInputDir: str = "/home/alextay96/Deep_Learning_Data/raw_src"
    writeOutputDir: str = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data"
    caseDfPath: str = (
        "/home/alextay96/Deep_Learning_Data/raw_src/complete_encoded_no_limit.parquet"
    )
    fileDfpath: str = "/home/alextay96/Deep_Learning_Data/raw_src/file_metadata.parquet"
    datasetName: str = "dmg_vision_dataset"


cs = ConfigStore.instance()
cs.store(name="de_config", node=ConfigParams)

log = logging.getLogger(__name__)
log.setLevel("DEBUG")
