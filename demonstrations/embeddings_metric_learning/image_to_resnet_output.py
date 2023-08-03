# Requirements: Download the dataset from https://download.pytorch.org/tutorial/hymenoptera_data.zip and extract in the subfolder /hymenoptera_data.

import os
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
import numpy as np

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

class Identity(torch.nn.Module):
    """Dummy layer to replace the prediction layer of ResNet.
    This is needed because we want to repurpose the model for a different
    task than it was initially trained on."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


data_dir = "hymenoptera_data"
image_datasets = {
    x if x == "train" else "validation": ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]
    )
    for x in ["train", "val"]
}


# initialize dataloader
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False)
    for x in ["train", "validation"]
}


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False


# delete the last layer of ResNet18
model.fc = Identity()

# set to evaluation mode
model.eval()

X_train = []
Y_train = []
for inputs, labels in dataloaders["train"]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    X_train.append(outputs[0].numpy())
    Y_train.append(labels[0].numpy())

np.savetxt("X_antbees.txt", np.array(X_train))
np.savetxt("Y_antbees.txt", np.array(Y_train))

X_test = []
Y_test = []
for inputs, labels in dataloaders["validation"]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    X_test.append(outputs[0].numpy())
    Y_test.append(labels[0].numpy())

np.savetxt("X_antbees_test.txt", np.array(X_test))
np.savetxt("Y_antbees_test.txt", np.array(Y_test))
