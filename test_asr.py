import gc
import pickle
import time
from torchvision.models.inception import inception_v3
from glob import glob
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
import torchvision.transforms as tfs
from torchvision import datasets, transforms
import torchvision
import re
import torchvision.transforms as transforms
from PIL import Image
import argparse
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--model", "-m", default=1, type=int, choices=[1, 2, 3, 4])
args = parser.parse_args()
# with open("/home/cw3344@drexel.edu/Semantic_Adversarial_Attacks/CelebA-HQ-to-CelebA-mapping.txt", "r") as f:
#     hq_to_celeba = [line.rstrip() for line in f]
# hq_to_celeba = hq_to_celeba[1:]
custom_transforms = transforms.Compose(
    [
        # transforms.CenterCrop(178),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def process_hq_data():
    with open(
        "/home/cw3344@drexel.edu/Semantic_Adversarial_Attacks/CelebA-HQ-to-CelebA-mapping.txt",
        "r",
    ) as f:
        hq_to_celeba = [line.rstrip() for line in f]
    hq_to_celeba = hq_to_celeba[1:]
    custom_transforms = transforms.Compose(
        [
            # transforms.CenterCrop(178),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    data_dir = "CelebA_HQ_facial_identity_dataset"
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), custom_transforms
    )
    # test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
    # save_path = "facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"
    # desc = "face identity classification"
    with open("saved_indices/saved_indices.pkl", "rb") as f:
        saved_indices = pickle.load(f)
    source_index_train = saved_indices["source_index_train"]
    target_index_train = saved_indices["target_index_train"]
    source_index_test = saved_indices["source_index_test"]
    target_index_test = saved_indices["target_index_test"]
    while True:
        # source_inputs, source_labels = train_dataset[source_index_train[step]]
        # target_inputs, target_labels = train_dataset[target_index_train[step]]
        yield lambda step: train_dataset[source_index_train[step]]


data_dir = "CelebA_HQ_facial_identity_dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), custom_transforms)
# test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
# desc = "face identity classification"
with open("saved_indices/saved_indices.pkl", "rb") as f:
    saved_indices = pickle.load(f)
source_index_train = saved_indices["source_index_train"]
target_index_train = saved_indices["target_index_train"]
source_index_test = saved_indices["source_index_test"]
target_index_test = saved_indices["target_index_test"]
device = torch.device(f"cuda")


if args.model == 1:
    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features, 307
    )  # multi-class classification (num_of_class == 307)
    # save_path = "facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"
    save_path = "models_celebhq/facial_identity_classification_using_transfer_learning_with_ResNet18_resolution_256_normalize_05.pth"
elif args.model == 2:
    model = torchvision.models.densenet121(pretrained=True)
    num_features = model.classifier.in_features
    model.fc = nn.Linear(num_features, 307)
    save_path = "models_celebhq/facial_identity_classification_using_transfer_learning_with_DenseNet121_resolution_256_normalize_05.pth"
elif args.model == 3:
    model = models.mnasnet1_0(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        num_features, 307
    )  # multi-class classification (num_of_class == 307)
    save_path = "models_celebhq/facial_identity_classification_using_transfer_learning_with_MNASNet_resolution_256_normalize_05.pth"
elif args.model == 4:
    model = models.resnet101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features, 307
    )  # multi-class classification (num_of_class == 307)
    save_path = "models_celebhq/facial_identity_classification_using_transfer_learning_with_ResNet101_resolution_256_normalize_05.pth"
model.to(device)

model.load_state_dict(torch.load(save_path, map_location=torch.device(f"cuda")))

criterion = nn.CrossEntropyLoss()
model.eval()
start_time = time.time()

ori_path = "filter_ori"
rec_path = "filter_rec"
folders = sorted(os.listdir(rec_path))
real_data_generator = process_hq_data()
with torch.no_grad():
    for folder in tqdm(folders):
        model.eval()
        test_loss = 0
        success = 0
        total = 0
        print("dataset:", folder)
        files = sorted(os.listdir(os.path.join(rec_path, folder)))
        for f in tqdm(files):
            if len(re.findall(r"\d+", f)[0]) != 0:
                id = int(re.findall(r"\d+", f)[0])
                image = custom_transforms(
                    Image.open(os.path.join(rec_path, folder, f)).convert("RGB")
                ).unsqueeze(0)
                image2, label = next(real_data_generator)(id)
                image = image.to(device)
                label = torch.tensor([label], dtype=torch.long).to(device)
                outputs = model(image)
                loss = criterion(outputs, label)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += 1
                success += 1 if predicted.item() != label.item() else 0
        print(
            f"Loss: {test_loss / (total):.4f} | ASR: {100.0 * success/total:.4f} % | ({success}/{total})"
        )
# for step in tqdm(range(len(source_index_train))):
#     source_inputs, source_labels = train_dataset[source_index_train[step]]
#     target_inputs, target_labels = train_dataset[target_index_train[step]]
#     hq_path = train_dataset.imgs[source_index_train[step]][0]
#     hq_label = train_dataset.imgs[source_index_train[step]][1]
# hq_id = int(hq_path.split("/")[-1].replace(".jpg", ""))
# with torch.no_grad():
#     running_loss = 0.0
#     running_corrects = 0

#     for i, (inputs, labels) in enumerate(test_dataloader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)

#         running_loss += loss.item() * inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data)

#         if i == 0:
#             print("[Original Image Examples]")
#             images = torchvision.utils.make_grid(inputs[:4])
#             imshow(images.cpu(), title=[class_names[x] for x in labels[:4]])
#             images = torchvision.utils.make_grid(inputs[4:8])
#             imshow(images.cpu(), title=[class_names[x] for x in labels[4:8]])
#             print("[Prediction Result Examples]")
#             images = torchvision.utils.make_grid(inputs[:4])
#             imshow(images.cpu(), title=[class_names[x] for x in preds[:4]])
#             images = torchvision.utils.make_grid(inputs[4:8])
#             imshow(images.cpu(), title=[class_names[x] for x in preds[4:8]])

#     epoch_loss = running_loss / len(test_dataset)
#     epoch_acc = running_corrects / len(test_dataset) * 100.0
#     print(
#         "[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s".format(
#             epoch, epoch_loss, epoch_acc, time.time() - start_time
#         )
#     )

# [Train #29] Loss: 0.0028 Acc: 99.9531% Time: 1387.7820s                                                                │
# [Test #29] Loss: 0.4981 Acc: 89.6296% Time: 1398.2725s
