import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RepeatedKFold,GridSearchCV
from scipy.stats import pearsonr,ttest_ind,levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA

fp = '/Users/mac/Documents/clinical.xlsx'
data = pd.read_excel(fp)
data = shuffle(data)
data = data.fillna(0)
data_1 = data.loc[data['Label'].isin([0])] #筛选出Label值为0的行
data_2 = data.loc[data['Label'].isin([1])]

X = data[data.columns[1:]]
y = data['Label']
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
colNames = X.columns #

'''PCA'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
model_pca = PCA(n_components=0.99) #涵盖99%的信息
model_pca.fit(X_train) #转换后的X_train矩阵
print(model_pca.explained_variance_)
print(model_pca.explained_variance_ratio_)

'''svm'''
X_train_pca = model_pca.transform(X_train)
X_test_pca = model_pca.transform(X_test)
print(X_train_pca.shape,X_test_pca.shape)
model_svm = svm.SVC(kernel='rbf',gamma='auto',probability=True).fit(X_train_pca,y_train)
score_svm = model_svm.score(X_test_pca,y_test)
print(score_svm)