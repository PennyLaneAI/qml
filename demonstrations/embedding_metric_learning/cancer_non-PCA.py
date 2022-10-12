from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(seed=123)

breast = load_breast_cancer()
breast_data = breast.data

breast_labels = breast.target

labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)

breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names

features_labels = np.append(features,'label')
breast_dataset.columns = features_labels

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)

x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)

feat_cols = ['feature'+str(i+1) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)

copy = normalised_breast.copy()
extracted_column = breast_dataset['label']
copy = copy.join(extracted_column)

b = copy[copy['label'] == 'Benign']
m = copy[copy['label'] == 'Malignant']

b['label'] = np.where(b['label'] == 'Benign', -1, 1)
m['label'] = np.where(m['label'] == 'Benign', -1, 1)

b = b.sample(frac=1).reset_index(drop=True)
m = m.sample(frac=1).reset_index(drop=True)

b_train_df = b.head(127)
b_test_df = b.tail(85)
m_train_df = m.head(214)
m_test_df = m.tail(143)

train_df = b_train_df.append(m_train_df)
test_df = b_test_df.append(m_test_df)

x_train_array = train_df.iloc[:,:-1].to_numpy()
x_test_array = test_df.iloc[:,:-1].to_numpy()
y_train_array = train_df[['label']].to_numpy()
y_test_array = test_df[['label']].to_numpy()

np.savetxt('bc_x_array.txt', x_train_array)
np.savetxt('bc_x_test_array.txt', x_test_array)
np.savetxt('bc_y_array.txt', y_train_array)
np.savetxt('bc_y_test_array.txt', y_test_array)
