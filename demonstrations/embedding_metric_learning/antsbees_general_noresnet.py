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

train_data = datasets.ImageFolder('hymenoptera_data/train',  data_transforms['transform'])
test_data = datasets.ImageFolder('hymenoptera_data/val',  data_transforms['transform'])
class_names = train_data.classes

Y_train = [1 if inp[1] == 0 else -1 for inp in train_data]
Y_val = [1 if inp[1] == 0 else -1 for inp in test_data]
Y_train = np.array(Y_train)
Y_train = np.reshape(Y_train, (244, 1))
Y_val = np.array(Y_val)
Y_val = np.reshape(Y_val, (153, 1))

X_train = []
for id, inp in enumerate(train_data):
    x = inp[0].detach().numpy()
    X_train.append(x)    
X_train = np.array(X_train)
X_train = np.reshape(X_train, (244, 224, 224, 3))
                   
X_val = []
for id, inp in enumerate(test_data):
    x = inp[0].detach().numpy()
    X_val.append(x)
X_val = np.array(X_val)
X_val = np.reshape(X_val, (153, 224, 224, 3))

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

X_tot_flat = X_tot.reshape(-1,150528)
feat_cols = ['pixel'+str(i) for i in range(X_tot_flat.shape[1])]

df_antbee = pd.DataFrame(X_tot_flat,columns=feat_cols)
df_antbee['label'] = Y_tot

pca_antbee = PCA(n_components=pc)
principalComponents_antbee = pca_antbee.fit_transform(df_antbee.iloc[:,:-1])

principal_antbee_Df = pd.DataFrame(data = principalComponents_antbee)
principal_antbee_Df.columns = ["PC"+str(i+1) for i in range(pc)]
principal_antbee_Df['label'] = Y_tot

principal_antbee_Df['label'].replace(1, 'Ant', inplace = True)
principal_antbee_Df['label'].replace(-1, 'Bee', inplace = True)

copy = principal_antbee_Df.copy()

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

print('Explained variation per principal component: {}'.format(pca_antbee.explained_variance_ratio_))

np.savetxt('ab_x_array.txt', x_train_array)
np.savetxt('ab_x_test_array.txt', x_test_array)
np.savetxt('ab_y_array.txt', y_train_array)
np.savetxt('ab_y_test_array.txt', y_test_array)
