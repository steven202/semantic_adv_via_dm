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
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG

from datasets.data_utils import get_dataset
import torchvision.transforms as tfs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device object

train_transform = tfs.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),  # data augmentation
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
data_dir = "./CelebA_HQ_facial_identity_dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)

print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

class_names = train_dataset.classes
# print('Class names:', class_names)

plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 60
plt.rcParams.update({"font.size": 20})


def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()


# load a batch of train image
iterator = iter(train_dataloader)

# visualize a batch of train image
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
imshow(out, title=[class_names[x] for x in classes[:4]])
# model = ResNet18(num_classes=307)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_features, 256), nn.Linear(256, 307))  # multi-class classification (num_of_class == 307)
model = model.to(device)
# quit()
if True:  # device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 30
start_time = time.time()

for epoch in tqdm(range(num_epochs)):
    """Training Phase"""
    model.train()

    running_loss = 0.0
    running_corrects = 0

    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # get loss value and update the network weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.0
    print("[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s".format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    """ Test Phase """
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.0
        print("[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s".format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
    save_path = "facial_identity_classification_transfer_learning_with_ResNet18_resolution_256_extractor.pth"
    torch.save(model.module.state_dict(), save_path)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 307)  # multi-class classification (num_of_class == 307)
model.load_state_dict(torch.load(save_path))
model.to(device)

model.eval()
start_time = time.time()

with torch.no_grad():
    running_loss = 0.0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if i == 0:
            print("[Original Image Examples]")
            images = torchvision.utils.make_grid(inputs[:4])
            imshow(images.cpu(), title=[class_names[x] for x in labels[:4]])
            images = torchvision.utils.make_grid(inputs[4:8])
            imshow(images.cpu(), title=[class_names[x] for x in labels[4:8]])
            print("[Prediction Result Examples]")
            images = torchvision.utils.make_grid(inputs[:4])
            imshow(images.cpu(), title=[class_names[x] for x in preds[:4]])
            images = torchvision.utils.make_grid(inputs[4:8])
            imshow(images.cpu(), title=[class_names[x] for x in preds[4:8]])

    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects / len(test_dataset) * 100.0
    print("[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s".format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

# [Train #29] Loss: 0.0028 Acc: 99.9531% Time: 1387.7820s                                                                │
# [Test #29] Loss: 0.4981 Acc: 89.6296% Time: 1398.2725s
# CUDA_VISIBLE_DEVICES=1 python train_classifier_extractor.py >> classifier_logs/train_classifier_extractor.txt
