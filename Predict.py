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
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    confusion_matrix,
)


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


def GetMultilabelPred():
    with open("./data/angle.json", "r") as f:
        anglePartMap = json.load(f)
    outputDir = pathlib.Path("./tmp/multilabel_pred_strat")
    os.makedirs(outputDir, exist_ok=True)
    for view, parts in tqdm(anglePartMap.items(), desc="angle"):
        runName = f"cv_pred_{view}"
        query = f"tags.`mlflow.runName`='{runName}'"
        runs = MlflowClient().search_runs(
            experiment_ids=["68"],
            filter_string=query,
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        info = runs[0].info
        runId = info.run_id
        mlflow.artifacts.download_artifacts(run_id=runId, dst_path=outputDir)


def ReadMultiLabelDf():
    search = "./tmp/multilabel_pred_strat/**.csv"
    allPredfile = glob.glob(search, recursive=True)
    allDf = dict()
    for p in allPredfile:
        imgAngle = p.split("/")[-1].split(".")[0]
        df = pd.read_csv(p)
        df["file"] = df["file"].apply(lambda x: ast.literal_eval(x)[0])
        df["CaseID"] = df["file"].apply(lambda x: int(x.split("/")[-1].split("_")[0]))
        allDf[imgAngle] = df
        # print(df[["CaseID"]])
    return allDf


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


def getAllPart():

    allPartPath = "./data/all_part.json"
    assert os.path.exists(allPartPath)
    with open(allPartPath, "r") as f:
        allParts = json.load(f)
    return allParts


def EnsembleMultilabel(allViewDf: dict[str, pd.DataFrame]):
    allParts = getAllPart()
    allCaseId = GetCasesWithEssentialAngles()
    allCasePred = []
    allSubsetAcc = []
    for pId, caseId in enumerate(tqdm(allCaseId, desc="Case")):
        casePred = {"CaseID": caseId}
        acc = 0
        for part in allParts:
            predFromDiffView = []
            gt = None
            for view, viewDf in allViewDf.items():
                partname = f"pred_{part}"
                if partname in viewDf.columns:
                    targetRowData = viewDf[(viewDf["CaseID"] == caseId)]
                    if targetRowData.empty:
                        continue
                    # print(targetRowData)
                    targetStatus = targetRowData[partname].item()

                    predFromDiffView.append(targetStatus)
                    if gt is None:
                        gt = viewDf[(viewDf["CaseID"] == caseId)][f"gt_{part}"].item()

            if len(predFromDiffView) == 0:
                # print(f"Skipped : {caseId}")
                predFromDiffView.append(False)

            ensemblePred = Counter(predFromDiffView).most_common(2)

            allCount = [x[1] for x in ensemblePred]

            if len(allCount) > 1 and allCount[0] == allCount[1]:
                casePred[part] = True
            else:
                casePred[part] = ensemblePred[0][0]
            if gt == casePred[part]:
                acc += 1
        subsetAcc = acc / len(allParts)
        allSubsetAcc.append(subsetAcc)
        allCasePred.append(casePred)
        if pId % 500 == 0:
            print(np.mean(allSubsetAcc))
    print(np.mean(allSubsetAcc))
    multilabelPredDf = pd.json_normalize(allCasePred)
    multilabelPredDf.to_csv("./tmp/multilabel_pred_strat.csv")


def EvalMultilabel():
    predDf = pd.read_csv("./tmp/multilabel_pred_strat.csv")
    gtDf = pd.read_csv(
        "/home/alextay96/Desktop/new_workspace/partlist_prediction/data/processed/best_2/multilabel_2.csv"
    )
    gtDf = gtDf[gtDf["CaseID"].isin(predDf["CaseID"].unique().tolist())]
    allParts = getAllPart()
    allEvalResults = []
    allPredByPart = {x: [] for x in allParts}
    allGtByPart = {x: [] for x in allParts}
    allTPByPart = {x: 0 for x in allParts}
    allTNByPart = {x: 0 for x in allParts}

    for rowId, (_, row) in enumerate(tqdm(predDf.iterrows())):
        caseId = row["CaseID"]
        gtRow = gtDf[gtDf["CaseID"] == caseId]
        acc = 0
        payload = {"CaseID": caseId}
        for part in allParts:
            gtPart = True if gtRow[f"vision_{part}"].item() == 1 else False
            predPart = row[part]
            if np.isnan(predPart):
                raise Exception("")
            allPredByPart[part].append(predPart)
            allGtByPart[part].append(gtPart)

            if gtPart == predPart:
                payload[part] = 1
                acc += 1
                if gtPart == True:
                    allTPByPart[part] += 1
                else:
                    allTNByPart[part] += 1
            else:
                payload[part] = 0

        caseAcc = acc / len(allParts)
        payload["subset_acc"] = caseAcc
        payload["correct"] = acc

        allEvalResults.append(payload)
        if rowId % 200 == 0:
            print(np.mean([x["subset_acc"] for x in allEvalResults]))
    evalresultDf = pd.json_normalize(allEvalResults)
    evalresultDf.to_csv("./tmp/multilabel_result_strat.csv")
    allMetrics = []
    for part in allParts:
        metricByPart = {"part": part}
        metricByPart["precision"] = average_precision_score(
            allGtByPart[part], allPredByPart[part]
        )
        metricByPart["recall"] = recall_score(allGtByPart[part], allPredByPart[part])
        metricByPart["tp"] = allTPByPart[part] / len(predDf)
        metricByPart["tn"] = allTNByPart[part] / len(predDf)
        metricByPart["acc"] = (allTPByPart[part] + allTNByPart[part]) / len(predDf)
        allMetrics.append(metricByPart)
    perfBreakdownDf = pd.json_normalize(allMetrics)
    perfBreakdownDf.to_csv("./tmp/multilabel_breakdown_strat.csv")


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
    # EvalCaseAcc()
    GetMultilabelPred()
    allViewDf = ReadMultiLabelDf()
    EnsembleMultilabel(allViewDf=allViewDf)
    EvalMultilabel()
