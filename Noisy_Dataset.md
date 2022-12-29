# Noisy label corrections

## Remove 20% of data

### 1 image has multiple labels. Removing the entire case based on 1 label will remove a large portion of cases

## Keep all data but reweight loss and eval metrics

### Need manual implementation to torchmetrics or need to convert all metrics to sklearn.metrics

## Change the label at the image multilabel dataset and maintain noisy label in part predictions

### Can keep all data and maintain good macro performance metrics

### No need to change training code

# Implementations
