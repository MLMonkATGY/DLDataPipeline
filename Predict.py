import mlflow
import ujson as json
from mlflow.tracking import MlflowClient
from ClassifierDataset import DmgClassifierDataset
from KFoldTrainPredict import ProcessModel
from data import ImportEnv
import pathlib
import os
from tqdm import tqdm
import glob
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader2
import torch
from torch.cuda.amp import autocast
import torchmetrics
import numpy as np
import itertools
import ast
from joblib import Parallel, delayed


def DownloadModels():
    with open(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/angle.json", "r"
    ) as f:
        anglePartMap = json.load(f)
    outputDir = pathlib.Path(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/models"
    )

    for view, parts in tqdm(anglePartMap.items(), desc="angle"):
        for part in tqdm(parts, desc="part"):
            outputModelDir = outputDir / f"{view}_{part}"
            os.makedirs(outputModelDir, exist_ok=True)

            runName = f"{view}_{part}_5"
            query = f"tags.`mlflow.runName`='{runName}'"
            runs = MlflowClient().search_runs(
                experiment_ids=["65"],
                filter_string=query,
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            info = runs[0].info
            runId = info.run_id
            allArtifacts = mlflow.artifacts.download_artifacts(
                run_id=runId, dst_path=outputModelDir
            )


def GetCasesWithEssentialAngles():
    allPaths = [
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Front View Left_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Front View Right_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Rear View Left_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Rear View Right_img_label.csv",
    ]
    perfectCaseDf = pd.read_csv(allPaths[0])
    for p in allPaths[1:]:
        df = pd.read_csv(p)
        perfectCaseDf = perfectCaseDf.merge(df, on="CaseID")
    perfectCaseId = perfectCaseDf["CaseID"].unique().tolist()
    return perfectCaseId


def GetAllCaseToProcess():
    allPaths = [
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Front View Left_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Front View Right_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Rear View Left_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Rear View Right_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Front View_img_label.csv",
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/train_test_labels/Saloon-4Dr_Rear View_img_label.csv",
    ]
    allDf = []
    for p in allPaths:
        df = pd.read_csv(p)
        allDf.append(df)

    processDf = pd.concat(allDf)
    return processDf


def LoadModel():
    srcDirPath = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/models"
    allModelName = glob.glob(srcDirPath + "/**", recursive=False)
    allModelNameWithPath = []
    for modelDir in allModelName:
        allCheckpoints = sorted(os.listdir(modelDir + "/checkpoints"))
        targetCkpt = os.path.join(modelDir + "/checkpoints", allCheckpoints[-1])
        modelName = modelDir.split("/")[-1]
        viewName = modelName.split("_")[0]
        partName = modelName.replace(viewName + "_", "")
        allModelNameWithPath.append(
            {
                "part_name": partName,
                "view_angle": viewName,
                "model_name": modelName,
                "model_path": targetCkpt,
            }
        )
    modelDf = pd.json_normalize(allModelNameWithPath)
    print(modelDf)
    return modelDf


def GeneratePrediction(modelDf: pd.DataFrame, processDf: pd.DataFrame):
    evalTransform = A.Compose(
        [
            A.LongestMaxSize(480),
            A.PadIfNeeded(
                min_height=480,
                min_width=480,
                border_mode=0,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    device = torch.device("cuda")

    accMetrics = torchmetrics.Accuracy(num_classes=2).to(device)
    predOutputCol = ["pred", "gt", "files", "part", "view", "model"]
    predOutputDf = pd.DataFrame([], columns=predOutputCol)
    for _, modelInfo in tqdm(modelDf.iterrows(), desc="model"):
        partName = modelInfo["part_name"]
        viewAngle = modelInfo["view_angle"]
        modelPath = modelInfo["model_path"]
        targetFilesDf = processDf[processDf["ViewAngle"] == viewAngle]
        ds = DmgClassifierDataset(
            "", colName=partName, targetDf=targetFilesDf, transform=evalTransform
        )
        evalLoader = DataLoader2(
            ds,
            shuffle=False,
            batch_size=100,
            num_workers=5,
        )
        model = ProcessModel.load_from_checkpoint(modelPath)
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            for batch in tqdm(evalLoader, desc="pred"):
                imgs = batch["img"].to(device)
                targets = batch["target"].numpy()
                files = batch["file"]
                with torch.autocast("cuda"):
                    output = model(imgs)
                    pred = torch.argmax(output, dim=1)
                    softmax = torch.softmax(output, dim=1)
                    (values, indices) = torch.max(softmax, dim=1)
                accMetrics.update(pred, batch["target"].to(device))
                predNp = pred.cpu().numpy()
                predProbNp = values.cpu().numpy()
                assertMask = predProbNp >= 0.5
                assert False not in assertMask
                predInfoList = []
                for p, g, f, prob in zip(
                    predNp,
                    targets,
                    files,
                    predProbNp,
                ):
                    predInfoList.append([p, g, f, prob])
                batchDf = pd.DataFrame(
                    predInfoList, columns=["pred", "gt", "files", "pred_prob"]
                )
                batchDf["part"] = partName
                batchDf["view"] = viewAngle
                batchDf["model"] = modelPath

                predOutputDf = pd.concat([predOutputDf, batchDf])
        batchAcc = accMetrics.compute()
        print(f"Current Acc : {batchAcc}")
    predOutputDf.to_csv("./raw_image_pred.csv")
    Acc = accMetrics.compute()
    print(f"Avg Acc : {Acc}")


def GetCrossValPred():
    with open(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/data/angle.json", "r"
    ) as f:
        anglePartMap = json.load(f)
    outputDir = pathlib.Path(
        "/home/alextay96/Desktop/new_workspace/DLDataPipeline/cross_val_pred"
    )

    for view, parts in tqdm(anglePartMap.items(), desc="angle"):
        for part in tqdm(parts, desc="part"):
            outputModelDir = outputDir / f"{view}_{part}"
            os.makedirs(outputModelDir, exist_ok=True)

            runName = f"cv_pred_{view}_{part}"
            query = f"tags.`mlflow.runName`='{runName}'"
            runs = MlflowClient().search_runs(
                experiment_ids=["65"],
                filter_string=query,
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            info = runs[0].info
            runId = info.run_id
            mlflow.artifacts.download_artifacts(run_id=runId, dst_path=outputModelDir)


def transform_pred(predDf):
    predDf["gt"] = list(itertools.chain(*predDf["gt"].apply(ast.literal_eval).tolist()))
    predDf["pred"] = list(
        itertools.chain(*predDf["pred"].apply(ast.literal_eval).tolist())
    )

    # predDf = predDf[predDf["gt"] == 1]
    allFilepath = np.array(
        list(itertools.chain(*predDf["file"].apply(ast.literal_eval).tolist()))
    )
    predDf["pred_probs"] = predDf["pred_probs"].apply(ast.literal_eval).apply(np.array)
    allStackProbs = np.stack(predDf["pred_probs"].values, 0)
    predProb = np.max(allStackProbs, axis=1)
    assert predProb.min() >= 0.5
    allGt = predDf["gt"].values
    allPred = predDf["pred"].values
    return predProb, allFilepath, allGt, allPred


def IntegrateCVPred():
    srcDir = "/home/alextay96/Desktop/new_workspace/DLDataPipeline/cross_val_pred"
    allCVPredPath = glob.glob(srcDir + "/**/*.csv", recursive=True)
    predOutputCol = ["pred", "gt", "filepath", "filename", "part", "view"]
    predOutputDf = pd.DataFrame([], columns=predOutputCol)
    for i in tqdm(allCVPredPath):
        fileElem = i.split("/")[-1].replace("cv_pred_", "").replace(".csv", "")
        view = fileElem.split("_")[0]
        part = fileElem.replace(f"{view}_", "")
        df = pd.read_csv(i)
        predProb, allFilepath, allGt, allPred = transform_pred(df)
        allFilename = [x.split("/")[-1] for x in allFilepath]
        cvPredDf = pd.DataFrame(
            {
                "gt": allGt,
                "pred": allPred,
                "filepath": allFilepath,
                "filename": allFilename,
                "pred_prob": predProb,
            }
        )
        cvPredDf["part"] = part
        cvPredDf["view"] = view
        predOutputDf = pd.concat([predOutputDf, cvPredDf])
    predOutputDf.to_csv("./tmp/all_cv_pred.csv")

    return predOutputDf


def GetCaseWithCompleteImg():
    allRawPred = pd.read_csv("./tmp/raw_image_pred.csv")
    perfectCaseId = GetCasesWithEssentialAngles()
    allRawPred["CaseID"] = allRawPred["files"].apply(
        lambda x: int(x.split("/")[-1].split("_")[0])
    )
    allRawPred["filename"] = allRawPred["files"].apply(lambda x: x.split("/")[-1])
    allValidCaseDf = allRawPred[allRawPred["CaseID"].isin(perfectCaseId)]
    assert len(allValidCaseDf) < len(allRawPred)
    return allValidCaseDf


def ReplacePredWithUnseen():
    validPredDf = GetCaseWithCompleteImg()
    allCvPred = pd.read_csv("./tmp/all_cv_pred.csv")
    allCvPred["CaseID"] = allCvPred["filename"].apply(lambda x: int(x.split("_")[0]))

    # allPart = allCvPred["part"].unique().tolist()
    # allView = allCvPred["view"].unique().tolist()
    modified = 0
    remainRaw = 0
    unseenPredList = []

    for i, caseId in enumerate(tqdm(validPredDf["CaseID"].unique().tolist())):
        rawPredByCase = validPredDf[validPredDf["CaseID"] == caseId]
        rawUnseenCvPred = allCvPred[allCvPred["CaseID"] == caseId]
        for _, row in rawPredByCase.iterrows():
            rawpart = row["part"]
            rawView = row["view"]
            rawfilename = row["filename"]
            unseenPayload = {
                "CaseID": caseId,
                "part": rawpart,
                "view": rawView,
                "filename": rawfilename,
                "filepath": row["files"],
                "gt": row["gt"],
            }
            cvOption = rawUnseenCvPred[
                (rawUnseenCvPred["part"] == rawpart)
                & (rawUnseenCvPred["view"] == rawView)
                & (rawUnseenCvPred["filename"] == rawfilename)
            ]
            if not cvOption.empty:
                unseenPayload["pred"] = cvOption["pred"].item()
                unseenPayload["pred_prob"] = cvOption["pred_prob"].item()
                modified += 1
            else:
                unseenPayload["pred"] = row["pred"]
                unseenPayload["pred_prob"] = row["pred_prob"]
                remainRaw += 1
            unseenPredList.append(unseenPayload)
        if i % 500 == 0:
            print(f"Modified : {modified}")
            print(f"Raw : {remainRaw}")
            print(f"Modified Ratio : {(modified) / (remainRaw + modified)}")
    unseenPredDf = pd.json_normalize(unseenPredList)
    unseenPredDf.to_csv("./tmp/unseen_pred.csv")


def EnsembleWorker(caseIdList):
    predDf = pd.read_csv("./tmp/raw_image_pred.csv")
    predDf["CaseID"] = predDf["files"].apply(
        lambda x: int(x.split("/")[-1].split("_")[0])
    )

    predDf = predDf[predDf["CaseID"].isin(caseIdList)]
    allPart = predDf["part"].unique().tolist()

    allPartPredList = []
    for caseId in tqdm(set(caseIdList)):
        casePredInfo = {"CaseID": caseId}
        for part in allPart:
            partPred = predDf[(predDf["CaseID"] == caseId) & (predDf["part"] == part)]
            dmgStatus = 0
            if len(partPred) == 1:
                dmgStatus = partPred["pred"].item()
            # elif len(partPred) % 2 == 0:
            #     highestConPred = partPred[
            #         partPred["pred_prob"] == partPred["pred_prob"].max()
            #     ]
            #     # print(highestConPred)
            #     dmgStatus = highestConPred["pred"].item()
            elif len(partPred) > 1:
                # print(partPred["pred"])
                mostCommonPredDf = partPred["pred"].mode()
                if len(mostCommonPredDf) > 1:
                    mostCommonPred = 1
                else:
                    mostCommonPred = mostCommonPredDf.item()
                dmgStatus = mostCommonPred
            casePredInfo[f"pred_{part}"] = dmgStatus
        allPartPredList.append(casePredInfo)
    return allPartPredList


def EnsemblePred():
    predDf = pd.read_csv("./tmp/raw_image_pred.csv")
    predDf["CaseID"] = predDf["files"].apply(
        lambda x: int(x.split("/")[-1].split("_")[0])
    )

    allCaseId = predDf["CaseID"].unique().tolist()
    # allCaseId = allCaseId[:100]
    caseIdBatch = []
    batchSize = 2000
    worker = 10
    for i in range(0, len(allCaseId), batchSize):
        caseIdBatch.append(allCaseId[i : i + batchSize])
    allPartPredList = Parallel(n_jobs=worker)(
        delayed(EnsembleWorker)(batch) for batch in caseIdBatch
    )
    allPartPredList = list(itertools.chain(*allPartPredList))
    # allView = predDf["view"].unique().tolist()
    #
    predCompleteDf = pd.json_normalize(allPartPredList)
    print(predCompleteDf)
    predCompleteDf.to_csv("./tmp/complete_pred.csv")


def EvalCaseAcc():
    gtDf = pd.read_csv(
        "/home/alextay96/Desktop/new_workspace/partlist_prediction/data/processed/best_2/multilabel_2.csv"
    )
    predDf = pd.read_csv("./tmp/complete_pred.csv")
    gtDf = gtDf[gtDf["CaseID"].isin(predDf["CaseID"].unique().tolist())]

    evalPerfBreakdown = []
    targetCol = [x.replace("pred_", "") for x in predDf.columns if "pred_" in x]
    allAcc = []
    for caseId in tqdm(predDf["CaseID"].unique().tolist()):
        correct = 0
        wrong = 0
        wrongPart = []
        correctPart = []
        for part in targetCol:
            predFromImg = predDf[predDf["CaseID"] == caseId][f"pred_{part}"].item()
            gtFromPartList = gtDf[gtDf["CaseID"] == caseId][f"vision_{part}"].item()
            if predFromImg == gtFromPartList:
                correct += 1
                correctPart.append(part)
            else:
                wrong += 1
                wrongPart.append(part)
        caseAcc = correct / (correct + wrong)
        evalPerfBreakdown.append(
            {
                "CaseID": caseId,
                "caseAcc": caseAcc,
                "wrong_part": wrongPart,
                "correct_part": correctPart,
            }
        )
        allAcc.append(caseAcc)
        if len(evalPerfBreakdown) % 1000 == 0:
            print(np.mean(allAcc))
    perfBreakDownDf = pd.json_normalize(evalPerfBreakdown)
    perfBreakDownDf.to_csv("./tmp/complete_perf_by_case.csv")
    print(perfBreakDownDf["caseAcc"].mean())


if __name__ == "__main__":
    # DownloadModels()
    # GetCasesWithEssentialAngles()
    # modelDf = LoadModel()
    # fileDf = GetAllCaseToProcess()
    # GeneratePrediction(modelDf=modelDf, processDf=fileDf)
    # GetCrossValPred()

    # IntegrateCVPred()
    # ReplacePredWithUnseen()
    # EnsemblePred()
    EvalCaseAcc()
