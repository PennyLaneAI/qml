import torch
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd

np.random.seed(seed=123)

pc = 4 # change this value to vary the number of principal components

data_transforms = {
    'transform': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

train_data = datasets.ImageFolder('hymenoptera_data/train',  data_transforms['transform'])
test_data = datasets.ImageFolder('hymenoptera_data/val',  data_transforms['transform'])
class_names = train_data.classes

model = models.resnet18(pretrained=True)

model.fc = Identity()

Y_train = [1 if inp[1] == 0 else -1 for inp in train_data]
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

X_tot = []
for i in X_train:
    X_tot.append(i)
for j in X_val:
    X_tot.append(j)

Y_tot = []
for i in Y_train:
    Y_tot.append(i)
for j in Y_val:
    Y_tot.append(j)

X_tot = np.array(X_tot)
Y_tot = np.array(Y_tot)

labels = np.reshape(Y_tot,(397,1))
final_data = np.concatenate([X_tot,labels],axis=1)

dataset = pd.DataFrame(final_data)
features = []
for i in range(512):
    features.append("resnet_feature_"+str(i + 1))
features_labels = np.append(features, 'label')

dataset.columns = features_labels
dataset['label'].replace(1, 'Ant', inplace = True)
dataset['label'].replace(-1, 'Bee', inplace = True)

x = dataset.loc[:,features].values
x = StandardScaler().fit_transform(x)

feat_cols = ['feature'+str(i+1) for i in range(x.shape[1])]
normalised_dataset = pd.DataFrame(x,columns=feat_cols)

pca_dataset = PCA(n_components = pc)
principalComponents_dataset = pca_dataset.fit_transform(x)

principal_dataset_df = pd.DataFrame(data = principalComponents_dataset)
principal_dataset_df.columns = ["PC"+str(i+1) for i in range(pc)]

copy = principal_dataset_df.copy()
extracted_column = dataset['label']
copy = copy.join(extracted_column)

a = copy[copy['label'] == 'Ant']
b = copy[copy['label'] == 'Bee']

a['label'] = np.where(a['label'] == 'Bee', -1, 1)
b['label'] = np.where(b['label'] == 'Bee', -1, 1)

a = a.sample(frac=1).reset_index(drop=True)
b = b.sample(frac=1).reset_index(drop=True)

a_train_df = a.head(116)
a_test_df = a.tail(77)
b_train_df = b.head(122)
b_test_df = b.tail(82)

train_df = a_train_df.append(b_train_df)
test_df = a_test_df.append(b_test_df)

x_train_array = train_df.iloc[:,:-1].to_numpy()
x_test_array = test_df.iloc[:,:-1].to_numpy()
y_train_array = train_df[['label']].to_numpy()
y_test_array = test_df[['label']].to_numpy()

print('Explained variation per principal component: {}'.format(pca_dataset.explained_variance_ratio_))

np.savetxt('ab_x_array.txt', x_train_array)
np.savetxt('ab_x_test_array.txt', x_test_array)
np.savetxt('ab_y_array.txt', y_train_array)
np.savetxt('ab_y_test_array.txt', y_test_array)
