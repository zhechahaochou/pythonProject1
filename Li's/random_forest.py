import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RepeatedKFold,GridSearchCV
from scipy.stats import pearsonr,ttest_ind,levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

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

'''svm支持向量机分类（参数优化前）'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model_svm = svm.SVC(kernel='rbf',gamma='auto',probability=True).fit(X_train,y_train)
score_svm = model_svm.score(X_test,y_test)
print(score_svm)
'''参数优化params opt:svm'''
#一堆C，在2^-1到2^3中取10个（base不写默认底为10）
Cs = np.logspace(-1,3,10,base = 2)
#一堆γ
gammas = np.logspace(-4,1,50,base = 2)
param_grid = dict(C = Cs, gamma = gammas)
#cv=10  10则交叉验证，可以省略，函数会自动给一个值
grid = GridSearchCV(svm.SVC(kernel='rbf'),param_grid = param_grid,cv=10).fit(X,y)
print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']
'''svm支持向量机分类（参数优化后）'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model_svm = svm.SVC(kernel='rbf',gamma=0.08901572837528345,C = 0.9258747122872903,probability=True).fit(X_train,y_train)
score_svm = model_svm.score(X_test,y_test)
print(score_svm)

'''p次k折交叉验证'''
#k折交叉验证 10折 kf = KFold(10)
#2次3折交叉验证
rkf = RepeatedKFold(n_repeats=2,n_splits=3)
#以rkf的交叉验证方法来拆分X中的原始数据
for train_index,test_index in rkf.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model_svm = svm.SVC(kernel='rbf',C=C,gamma = gamma,probability=True).fit(X_train,y_train)
    score_svm = model_svm.score(X_test,y_test)
    # print('2次3折交叉验证：')
    print(score_svm)
#k折交叉验证的另一种方法（没法洗牌）
#score_svm = cross_val_score(model_svm,X,y,cv=3,scoring='accuracy')