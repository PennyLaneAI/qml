import os
import torch
from torchvision import datasets, models, transforms
import numpy as np

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalize input channels using mean values and standard deviations of ImageNet.
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
    x if x == "train" else "validation": datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
class_names = image_datasets["train"].classes

# Initialize dataloader
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True)
    for x in ["train", "validation"]
}

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.eval()

# Delete the last layer of ResNet18
model.fc = Identity()

X_train = []
Y_train = []
for inputs, labels in dataloaders["train"]:
    outputs = model(inputs)
    X_train.append(outputs[0].numpy())
    Y_train.append(labels[0].numpy())

np.savetxt("X_antbees_NEW.txt", np.array(X_train))
np.savetxt("Y_antbees_NEW.txt", np.array(Y_train))

X_test = []
Y_test = []
for inputs, labels in dataloaders["validation"]:
    outputs = model(inputs)
    X_test.append(outputs[0].numpy())
    Y_test.append(labels[0].numpy())

np.savetxt("X_antbees_test_NEW.txt", np.array(X_test))
np.savetxt("Y_antbees_test_NEW.txt", np.array(Y_test))
