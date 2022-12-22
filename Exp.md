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

src/train/part_per_model_sim.ipynb

## 6. Transfer data to local and updates local file metadata data/imgs_metadata/Saloon - 4 Dr.parquet

src/data_engineering/GetImgsDf.ipynb

## 7. Generate image label df for each image view

src/feature/MultiviewMultilabelDf.ipynb

## 8. Train multilabel image classification with new exp name

src/train/ImgTrainer.py

## 9. Ensemble predictions and evaluate part metrics

src/analysis/emsemble.py

## 10. Performance Analysis

src/analysis/Analyse.ipynb

# Experiments

## Hill climb performance

Get all part precision recall to optimal tradeoff
Prevent extreme low precision or extreme low recall

### Loss function

1. BCE
2. Focal loss
3. ASL
   Selected : Focal loss

### Image resolutions

1. 300
2. 480
3. 640
   Selected 480

### Pos weight

1. Without pos weight
2. With pos Weight
   Selected : Without

### Data Augmentations

1. Light
2. Light + Grid shuffle
   Selected : Light

### Training Epoch

1. 5
2. 10
   Selected : 5

### Ensemble default when even predictions

1. Default dmg
2. Default not dmg
3. By avg conf

### Model Size

## Catboost classifier

### Add ROC threshold

1. With threshold
2. Without threshold
   Selected : Without

### Iterations

1. 100
2. 1000

### Add features

1. Vehicle Model
2. Vehicle still driveable
3. NCB Stat
4. Claim type
5. Vehicle Type
6. Assembly_Type
7. Sum_Insured
8. Repairer
9. Repairer_Apprv_Count
10. Collision_With
11. Handling_Insurer
