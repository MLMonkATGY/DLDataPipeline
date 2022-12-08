# Process

# 0. Generate scope dataframe - Determining the filtering criteria for case, part and files

src/feature/Scope.ipynb

## 1. Generate fuzzy match partlist cluster

src/feature/PartlistLabel.ipynb

## 2. Update data/tmp/complete_view_mapping.csv for part clustering and part-to-view-mapping

## 3. Generate mapping for partlist

src/feature/PartlistLabel.ipynb

## 4. Genereate multilabel df for partlist prediction systems

src/feature/GenPartlistMultilabelDf.py

## 5. Train Catboost Multilabel System

src/train/PartlistPredition.ipynb

## 6. Transfer data to local and updates local file metadata data/imgs_metadata/Saloon - 4 Dr.parquet

src/data_engineering/GetImgsDf.ipynb

## 7. Generate image label df for each image view

src/feature/MultiviewMultilabelDf.ipynb

## 8. Train multilabel image classification

src/train/ImgTrainer.py

## 9. Ensemble predictions and evaluate part metrics

src/analysis/Predict.py

## 10. Performance Analysis

src/analysis/Analyse.ipynb

# Experiments

## Optimization objective

1. Part Avg TP
2. Part Avg TN
3. Subset Acc

## Optimization element order

### Loss Function (Focal vs BCE)

BCE

### Image Input Size (300 vs 640)

300

### Data Augmentation (Present vs Absent)

Absent

### Epoch (5 vs 10 vs 15)

### Model Sizes (B0 vs B3)

### Training Batch (50 vs 200 vs all)

Training batch has to be all to account for all label comb

### Pos Weight Scaler (1 vs 10)
