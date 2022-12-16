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

src/analysis/Predict.py

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
