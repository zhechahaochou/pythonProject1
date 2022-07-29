import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, GridSearchCV
from scipy.stats import pearsonr, ttest_ind, levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

fp = '/Users/mac/Documents/yan/results.xlsx'
data = pd.read_excel(fp)
data = data.fillna(0)

data_1 = data.loc[data['label_all'].isin([0])]  # 没病
data_2 = data.loc[data['label_all'].isin([1])]

'''t test'''
index = []
for colName in data.columns[2:]:
    if levene(data_1[colName], data_2[colName])[1] > 0.05:
        if ttest_ind(data_1[colName], data_2[colName])[1] < 0.05:
            index.append(colName)
    else:
        if ttest_ind(data_1[colName], data_2[colName], equal_var=False)[1] < 0.05:
            index.append(colName)
print('T test picks ' + str(len(index)))

'''lasso特征筛选'''
if 'label_all' not in index:index = ['label_all']+index
data_1 = data_1[index]
data_2 = data_2[index]
data = pd.concat([data_1, data_2])
data = shuffle(data)
data.index = range(len(data)) #打乱后重新标号

X = data[data.columns[2:]]
y = data['label_all']
X = X.apply(pd.to_numeric, errors='ignore') #将数据类型转化为数值型
colNames = X.columns
# X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames
# data_new = pd.merge(y, X, how='inner', left_index=True, right_index=True)  # 只是输出t检验和标准化后的data表
# data_new.to_excel('/Users/mac/Documents/yan/results_new.xlsx')

alphas = np.logspace(-3, 1, 50)
model_lassoCV = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(X, y)
print('The best γ is ' + str(model_lassoCV.alpha_))
coef = pd.Series(model_lassoCV.coef_, index=X.columns)
print("Lasso picked "+str(sum(coef!=0)) + "variables and eliminated the other "+str(sum(coef==0)))

index = coef[coef !=0].index
X = X[index]
print(coef[coef != 0])  #查看经过t检验和lasso的特征值

'''randomForest随机森林'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_rf = RandomForestClassifier(random_state=10, n_estimators=550).fit(X_train, y_train)
#效果查看
score_rf = model_rf.score(X_test, y_test)
print('rf score = ' + str(score_rf))

'''svm支持向量机分类（参数优化前）'''
model_svm = svm.SVC(kernel='rbf', gamma='auto', probability=True).fit(X_train, y_train)
score_svm = model_svm.score(X_test, y_test)
print('svm score = ' + str(score_svm))

''' 参数优化params opt:svm '''
# 一堆C，在2^-1到2^3中取10个（base不写默认底为10）
Cs = np.logspace(-1, 3, 10, base = 2)
# 一堆γ
gammas = np.logspace(-4, 1, 50, base=2)
param_grid = dict(C=Cs, gamma=gammas)
# cv=10  10则交叉验证，可以省略，函数会自动给一个值
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=10).fit(X_train, y_train)
print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']

'''svm支持向量机分类（参数优化后）'''
model_svm_2 = svm.SVC(kernel='rbf', gamma=0.10254191950095475, C=1.2599210498948732, probability=True).fit(X_train, y_train)
score_svm_2 = model_svm_2.score(X_test, y_test)
print(score_svm_2)


# # 学习曲线
# rfc_l = []
# for i in range(100):
#     rfc = RandomForestClassifier(n_estimators=i+1)
#     rfc_s = cross_val_score(rfc,X_test,y_test,cv=10).mean()
#     rfc_l.append(rfc_s)
# print(max(rfc_l),rfc_l.index(max(rfc_l)))  # 最高精确度取值，在森林数目为xx的时候得到
# plt.plot(range(1,201),rfc_l, label='randomforest')
# plt.legend()  # 显示图例
# plt.show()

