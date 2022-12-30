# Noisy label corrections

## Remove 20% of data

### 1 image has multiple labels. Removing the entire case based on 1 label will remove a large portion of cases

## Keep all data but reweight loss and eval metrics

### Need manual implementation to torchmetrics or need to convert all metrics to sklearn.metrics

## Change the label at the image multilabel dataset and maintain noisy label in part predictions

### Can keep all data and maintain good macro performance metrics

### No need to change training code

# Implementations

## Procedure to remove ood images

1. Get top 2000 cases image
2. Record OOD Images

## Procedure 2

1. Which label to invert ?
   bumper front, bumper rear, windscreen front, engine, rear_compartment, grille
2. How to determine target for label corrections ?
   Exclude misc and non external
   1. Pred = 0 and Gt = 1 for target label
   2. len(Gt) >=5 and len(Pred) == 1 and pred == bumper front or bumper rear
