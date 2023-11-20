import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt

import time
import os

from tqdm import tqdm
from configs.paths_config import (
    DATASET_PATHS,
    MODEL_PATHS,
    HYBRID_MODEL_PATHS,
    HYBRID_CONFIG,
)

from datasets.data_utils import get_dataset
import torchvision.transforms as tfs
import random

random.seed(0)


def shuffle_helper(lst):
    random.shuffle(lst)
    return lst


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device object

train_transform = tfs.Compose(
    [
        transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(),  # data augmentation
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]
)

test_transform = tfs.Compose(
    [
        transforms.Resize((256, 256)),
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]
)
data_dir = "./CelebA_HQ_face_gender_dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))
source_index_train = shuffle_helper(np.arange(len(train_dataset)))
target_index_train = shuffle_helper(np.arange(len(train_dataset)))
source_index_test = shuffle_helper(np.arange(len(test_dataset)))
target_index_test = shuffle_helper(np.arange(len(test_dataset)))

saved_indices = {
    "source_index_train": [],
    "target_index_train": [],
    "source_index_test": [],
    "target_index_test": [],
}

class_names = train_dataset.classes
# print('Class names:', class_names)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(
    num_features, len(train_dataset.classes)
)  # multi-class classification (num_of_class == 307)
model = model.to(device)
# quit()
if True:  # device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

save_path = "models_celebhq_gender/face_gender_classification_using_transfer_learning_with_ResNet18_resolution_256_normalize_05.pth"
model.module.load_state_dict(torch.load(save_path, map_location="cuda"))
"""train dataset"""
model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(source_index_train))):
        # if len(saved_indices["target_index_train"]) > 500:
        #     break
        source_inputs, source_labels = train_dataset[source_index_train[i]]
        target_inputs, target_labels = train_dataset[target_index_train[i]]
        source_inputs, target_inputs = source_inputs.unsqueeze(0).to(
            device
        ), target_inputs.unsqueeze(0).to(device)

        source_outputs, target_outputs = (
            model(source_inputs).max(dim=1)[1].item(),
            model(target_inputs).max(dim=1)[1].item(),
        )
        if (
            source_outputs == source_labels
            and target_outputs == target_labels
            and source_outputs != target_outputs
            and source_labels != target_labels
        ):
            saved_indices["source_index_train"].append(source_index_train[i])
            saved_indices["target_index_train"].append(target_index_train[i])
        else:
            pass

"""test dataset"""
model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(source_index_test))):
        # if len(saved_indices["target_index_test"]) > 500:
        #     break
        source_inputs, source_labels = test_dataset[source_index_test[i]]
        target_inputs, target_labels = test_dataset[target_index_test[i]]
        source_inputs, target_inputs = source_inputs.unsqueeze(0).to(
            device
        ), target_inputs.unsqueeze(0).to(device)

        source_outputs, target_outputs = (
            model(source_inputs).max(dim=1)[1].item(),
            model(target_inputs).max(dim=1)[1].item(),
        )
        if (
            source_outputs == source_labels
            and target_outputs == target_labels
            and source_outputs != target_outputs
            and source_labels != target_labels
        ):
            saved_indices["source_index_test"].append(source_index_test[i])
            saved_indices["target_index_test"].append(target_index_test[i])
        else:
            pass
for k, v in saved_indices.items():
    print(len(v))
with open("saved_indices/saved_indices_gender.pkl", "wb") as f:
    pickle.dump(saved_indices, f)
