import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RepeatedKFold,GridSearchCV
from scipy.stats import pearsonr,ttest_ind,levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import joblib

fp = '/Users/mac/Documents/clinical.xlsx'
data = pd.read_excel(fp)
data = shuffle(data)
data = data.fillna(0)
data_1 = data.loc[data['Label'].isin([0])] #筛选出Label值为0的行
data_2 = data.loc[data['Label'].isin([1])]

X = data[data.columns[1:]]
y = data['Label']
colNames = X.columns
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames #
'''t检验特征筛选'''
index = []
for colName in data.columns[1:]:
    if levene(data_1[colName],data_2[colName])[1] > 0.05:
        if ttest_ind(data_1[colName],data_2[colName])[1]<0.05:
            index.append(colName)
    else:
        if ttest_ind(data_1[colName],data_2[colName],equal_var=False)[1]<0.05:
            index.append(colName)
print(len(index))

'''lasso特征筛选'''
if 'Label' not in index:index = ['Label']+index
data_1 = data_1[index]
data_2 = data_2[index]
data = pd.concat([data_1,data_2])
data = shuffle(data)
data.index = range(len(data)) #打乱后重新标号
X= data[data.columns[1:]]
y = data['Label']
X = X.apply(pd.to_numeric, errors='ignore') #将数据类型转化为数值型
colNames = X.columns
X = X.fillna(0)
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames

alphas = np.logspace(-3,1,50)
model_lassoCV = LassoCV(alphas = alphas,cv=10,max_iter=100000).fit(X,y)
print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_,index = X.columns)
print("Lasso picked "+str(sum(coef!=0)) + "variables and eliminated the other "+str(sum(coef==0)))

index = coef[coef !=0 ].index
X = X[index]
print(coef[coef != 0])  #查看经过t检验和lasso的特征值

'''randomForest分类'''
#将筛选出的特征分为训练集和测试集，其中，测试集占30%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
#模型训练
model_rf = RandomForestClassifier().fit(X_train,y_train)
#效果查看
score_rf = model_rf.score(X_test,y_test)
print(score_rf)

'''在生成了随机森林模型后进行保存'''
#方一：导入joblib
#保存模型
modelPath = '/Users/mac/Documents/modelSave/rf.model'#模型存储路径
joblib.dump(model_rf, modelPath)#模型，模型存储路径
#调用模型
model_rf_load = joblib.load(modelPath)#取模型
score_rf_load = model_rf_load.score(X_test, y_test)#使用模型
print(score_rf_load)
