import torch
from torch import nn, optim
from baal.modelwrapper import ModelWrapper
from torchvision.models import vgg16
from baal.bayesian.dropout import MCDropoutModule
from baal.active import FileDataset, ActiveLearningDataset
from torchvision import transforms

from glob import glob
import os
from sklearn.model_selection import train_test_split

files = glob(
    "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/playground/natural_images/*/*.jpg"
)
classes = os.listdir(
    "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/playground/natural_images"
)
train, test = train_test_split(
    files, random_state=1337
)  # Split 75% train, 25% validation
print(f"Train: {len(train)}, Valid: {len(test)}, Num. classes : {len(classes)}")
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# We use -1 to specify that the data is unlabeled.
train_dataset = FileDataset(train, [-1] * len(train), train_transform)

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# We use -1 to specify that the data is unlabeled.
test_dataset = FileDataset(test, [-1] * len(test), test_transform)
active_learning_ds = ActiveLearningDataset(
    train_dataset, pool_specifics={"transform": test_transform}
)
USE_CUDA = torch.cuda.is_available()
model = vgg16(pretrained=False, num_classes=len(classes))
# This will modify all Dropout layers to be usable at test time which is
# required to perform Active Learning.
model = MCDropoutModule(model)
if USE_CUDA:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# ModelWrapper is an object similar to keras.Model.
baal_model = ModelWrapper(model, criterion)
from baal.active.heuristics import BALD

heuristic = BALD(shuffle_prop=0.1)
# This function would do the work that a human would do.
def get_label(img_path):
    return classes.index(img_path.split("/")[-2])


import numpy as np

# 1. Label all the test set and some samples from the training set.
for idx in range(len(test_dataset)):
    img_path = test_dataset.files[idx]
    test_dataset.label(idx, get_label(img_path))

# Let's label 100 training examples randomly first.
# Note: the indices here are relative to the pool of unlabelled items!
train_idxs = np.random.permutation(np.arange(len(train_dataset)))[:100].tolist()
labels = [get_label(train_dataset.files[idx]) for idx in train_idxs]
active_learning_ds.label(train_idxs, labels)

print(f"Num. labeled: {len(active_learning_ds)}/{len(train_dataset)}")
# 2. Train the model for a few epoch on the training set.
baal_model.train_on_dataset(
    active_learning_ds, optimizer, batch_size=16, epoch=5, use_cuda=USE_CUDA
)
baal_model.test_on_dataset(test_dataset, batch_size=16, use_cuda=USE_CUDA)

print("Metrics:", {k: v.avg for k, v in baal_model.metrics.items()})
pool = active_learning_ds.pool
if len(pool) == 0:
    raise ValueError("We're done!")

# We make 15 MCDropout iterations to approximate the uncertainty.
predictions = baal_model.predict_on_dataset(
    pool, batch_size=16, iterations=15, half=True, use_cuda=USE_CUDA, verbose=True
)
# We will label the 10 most uncertain samples.
top_uncertainty = heuristic(predictions)[:10]
print("Top uncertainty")
print(top_uncertainty)
# 4. Label those samples.
oracle_indices = active_learning_ds._pool_to_oracle_index(top_uncertainty)
labels = [get_label(train_dataset.files[idx]) for idx in oracle_indices]
print(list(zip(labels, oracle_indices)))
active_learning_ds.label(top_uncertainty, labels)
