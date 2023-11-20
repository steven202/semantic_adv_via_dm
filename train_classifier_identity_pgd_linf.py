import copy
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
from torchattacks import PGD, PGDL2


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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)

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
iterator = iter(train_loader)

# visualize a batch of train image
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
imshow(out, title=[class_names[x] for x in classes[:4]])
# model = ResNet18(num_classes=307)

net = models.resnet18(pretrained=True)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 307)  # multi-class classification (num_of_class == 307)
model = net.to(device)
# quit()
if True:  # device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model
        self.attack = PGD(model, eps=2 / 255, alpha=1 / 255, steps=10, random_start=True)

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x.requires_grad_()
        with torch.enable_grad():
            adv_image = self.attack(x, y)
        return adv_image


def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


learning_rate = 0.01
file_name = "resnet_pgd_linf"

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)


def train(epoch):
    print("\n[ Train epoch: %d ]" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print("\nCurrent batch:", str(batch_idx))
            print("Current adversarial train accuracy:", str(predicted.eq(targets).sum().item() / targets.size(0)))
            print("Current adversarial train loss:", loss.item())

    print("\nTotal adversarial train accuarcy:", 100.0 * correct / total)
    print("Total adversarial train loss:", train_loss)


def test(epoch):
    print("\n[ Test epoch: %d ]" % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print("\nCurrent batch:", str(batch_idx))
                print("Current benign test accuracy:", str(predicted.eq(targets).sum().item() / targets.size(0)))
                print("Current benign test loss:", loss.item())

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print("Current adversarial test accuracy:", str(predicted.eq(targets).sum().item() / targets.size(0)))
                print("Current adversarial test loss:", loss.item())

    print("\nTotal benign test accuarcy:", 100.0 * benign_correct / total)
    print("Total adversarial test Accuarcy:", 100.0 * adv_correct / total)
    print("Total benign test loss:", benign_loss)
    print("Total adversarial test loss:", adv_loss)

    state = {"net": net.state_dict()}
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(state, "./classifier_ckpts/" + file_name + ".pt")
    print("Model Saved!")


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


for epoch in tqdm(range(0, 200)):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
"""
rm classifier_logs/adv_linf.txt
CUDA_VISIBLE_DEVICES=2 python train_classifier_identity_pgd_linf.py >>  classifier_logs/adv_linf.txt
"""
