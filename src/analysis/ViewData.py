import glob
from typing import Dict, List
import pandas as pd
import os
import shutil
from tqdm import tqdm

from Config import ConfigParams
from pprint import pprint
import time


def readAndCombineDf(allLabelPath: List[str]):
    allDf = []
    for i in allLabelPath:
        df = pd.read_csv(i)
        allDf.append(df)
    return pd.concat(allDf)


def viewData(df: pd.DataFrame, allPart: List[str], cfg: ConfigParams):
    images_patt = "/path/to/images/*"
    removingPath = "/home/alextay96/.fiftyone/"
    if os.path.exists(removingPath):
        shutil.rmtree(removingPath)
    # Ex: your custom label format
    import fiftyone as fo

    import ujson as json
    import fiftyone.core.fields as fof
    from fiftyone.core.view import DatasetView
    import fiftyone.core.odm as foo
    samples = []

    # allSupportedparts = set([x for x in ])
    for _, row in tqdm(df.iterrows()):
        path = row["Path"]
        viewAngle = row["ViewAngle"]
        sample = fo.Sample(filepath=path)
        sample["view_angle"] = fo.Classification(
            label=viewAngle,
        )
        for p in allPart:
            dmgStatus = row[p]
            label = "dmg"
            if dmgStatus == 1:
                label = f"{p}_dmg"
            elif dmgStatus == 0:
                label = f"{p}_no_dmg"
            else:
                label = f"{p}_N/A"
            sample[p] = fo.Classification(
                label=label,
            )

        samples.append(sample)
    dataset = fo.Dataset(f"dmg_part_cls_dataset")
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    keyEntered = input()
    print("Key detected. Exiting app...")
    # session.wait(10)
 