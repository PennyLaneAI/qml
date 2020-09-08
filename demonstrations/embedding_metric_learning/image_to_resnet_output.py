import torch
from torchvision import datasets, models, transforms
import numpy as np

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

train_data = datasets.ImageFolder('hymenoptera_data/train',  data_transforms['train'])
test_data = datasets.ImageFolder('hymenoptera_data/val',  data_transforms['val'])
class_names = train_data.classes

model = models.resnet18(pretrained=True)
model.fc = Identity()

Y_train = [1 if inp[1] == 0 else -1 for inp in test_data]
Y_val = [1 if inp[1] == 0 else -1 for inp in test_data]

X_train = []
for id, inp in enumerate(train_data):
    inp_torch = torch.stack([inp[0]])
    x = model(inp_torch).detach().numpy()[0]
    X_train.append(x)

X_val = []
for id, inp in enumerate(test_data):
    inp_torch = torch.stack([inp[0]])
    x = model(inp_torch).detach().numpy()[0]
    X_val.append(x)

np.savetxt("X_antbees.txt", X_train)
np.savetxt("Y_antbees_train.txt", Y_train)
np.savetxt("X_antbees_test.txt", X_val)
np.savetxt("Y_antbees_test.txt", Y_val)
