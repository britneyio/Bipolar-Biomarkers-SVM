from sklearn import preprocessing
import pandas as pd
import openpyxl
from itertools import islice
import numpy as np
from matplotlib import pyplot as plt
df = openpyxl.load_workbook('biomarker_sheet.xlsx')

# Define variable to read sheet
ws = df['Supplemental Table 3']


# Unmerging the cells
for merge in list(ws.merged_cells):
    ws.unmerge_cells(range_string=str(merge))

# converting from excel sheet to dataframe so we can use it in python
data = ws.values
cols = next(data)[1:]
data = list(data)
idx = [r[0] for r in data]
data = (islice(r, 1, None) for r in data)
df = pd.DataFrame(data, index=idx, columns=cols)


df = df.dropna(how='all').dropna(axis=1, how='any')
df.reset_index(drop=True, inplace=True)

#Need to process the labels for the disorders
le = preprocessing.LabelEncoder()
le.fit(df['Disorder'])
df['Disorder'] = le.transform(df['Disorder'])

# standard scaler
# scaler = preprocessing.StandardScaler().fit(df)
# transform the data with the standard scalar
df_mean = df.mean()
df_std = df.std()
X = (df - df_mean) / df_std
#df_scaled = scaler.transform(df)

# number of data(N): 115
# number of features(m): 106
# N * m

# compute the covariance matrix of the data
cov_X = X.cov()

# perform eigendecomposition on the covariance matrix to find
# eigenvalues, and eigenvectors
eig_val, eig_vec = np.linalg.eig(cov_X)

# Index the eigenvalues in descending order
idx = eig_val.argsort()[::-1]
# Sort the eigenvalues in descending order
eig_val = eig_val[idx]
# sort the corresponding eigenvectors accordingly
eig_vec = eig_vec[:, idx]
total_var = cov_X.var().sum()
# explained_var = np.cumsum(eig_val)/total_var
explained_var = np.cumsum(eig_val) / np.sum(eig_val)
# choose the number of principal components to retain
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(explained_var, marker="o")
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Proportion of explained variance")
ax[0].set_title("Scree plot")

ax[1].plot(np.cumsum(explained_var), marker="o")
ax[1].set_xlabel("Principal component")
ax[1].set_ylabel("Cumulative sum of explained variance")
ax[1].set_title("Cumulative scree plot")
plt.show()
n_components = 30
# n_components = np.argmax(explained_var >= 0.5) + 1
print(n_components)
u = eig_vec[:,:n_components]
print(u.shape)

pc_list = []
for i in range(n_components):
    pc_list.append(i)
    
print(pc_list)

P = pd.DataFrame(u, index = df.columns, columns = pc_list)


#Then,  the standardized data matrix (X) by the matrix of selected principal components
# (P) to project the original data onto the selected principal components to obtain the reduced-
#dimensional representaGon of the data. So, you need to do: X*P

#project X on to P
projection = X @ P

#Split data into train adn test data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


y = df.Disorder

# x_train , x_test , y_train , y_test = train_test_split(projection , 
#                                                         y,
#                                                         test_size=0.3,
#                                                         random_state=101,
#                                                         stratify=y)

#Implement SVC and optomize parameters using Grid search
svc = svm.SVC()
kc_parameters = [0.001+i*0.005 for i in range(100)]

#Lists of different paraemters
parameters = {'kernel':['linear'],'C': kc_parameters}
opt_model = GridSearchCV(svc, parameters, return_train_score= True, scoring="accuracy")
opt_model.fit(projection,y) 
# print(opt_model.best_index_)
# print(opt_model.cv_results_)
#Get index for dictionary that summarizes parameters from best model
model_idx = opt_model.best_index_
model_accuracy = opt_model.best_score_
#retrieve best parameters and check testing, training, and accuracy score for overfitting
print("best parameters and score:" )
print(opt_model.cv_results_['mean_test_score'][model_idx])
print(opt_model.cv_results_['mean_test_score'])
print(opt_model.cv_results_['mean_train_score'][model_idx])
print(model_accuracy, opt_model.cv_results_['params'][model_idx])                                                    
# svm_pca = SVC(kernel = 'rbf',gamma = 0.01).fit(x_train,y_train)





