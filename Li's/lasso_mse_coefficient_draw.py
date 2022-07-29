import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
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
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
colNames = X.columns #
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

'''lasso作图'''
MSEs = (model_lassoCV.mse_path_)
#注释内容与下面两行代码效果一样
# MSEs_mean,MSEs_std = [],[]
# for i in range(len(MSEs)):
#     MSEs_mean.append(MSEs[i].mean())
#     MSEs_std.append(MSEs[i].std())
MSEs_mean = np.apply_along_axis(np.mean,1,MSEs) #1：取行，0：取列；MSEs：从这个矩阵里取内容
#标准差
MSEs_std = np.apply_along_axis(np.std,1,MSEs)
#作图（errorbar）
plt.figure(dpi=300) #作图的分辨率设为300（清晰）
plt.errorbar(model_lassoCV.alphas_,MSEs_mean #x轴数据：lambdaλ的值，y轴数据：均值
             ,yerr=MSEs_std #y误差范围
             ,fmt='o' #数据点标记
             ,ms=3 #数据点大小
             ,mfc='r' #数据点颜色
             ,mec='r' #数据点边缘颜色
             ,ecolor='lightblue' #误差棒颜色
             ,elinewidth=2 #误差棒线宽
             ,capsize=4 #误差棒边界线长度
             ,capthick=1) #误差棒边界线厚度
plt.semilogx() #x轴用对数形式
# axis vertical 竖直的线，表示取的最佳的alpha的值对应的线，用虚线表示
plt.axvline(model_lassoCV.alpha_,color='black',ls='--')
plt.xlabel('Lambda')
plt.ylabel('MSE')
ax=plt.gca()
y_major_locator=MultipleLocator(0.05) #y轴各参数间隔多少
ax.yaxis.set_major_locator(y_major_locator)
plt.show()
#作图 coefficients
# X_raw = model_lassoCV.alphas_
coefs = model_lassoCV.path(X, y, alphas=alphas, max_iter=100000)[1].T

plt.figure()
plt.semilogx(model_lassoCV.alphas_,coefs,'-') #用实线画每个特征随着参数的变化的值
# axis vertical 竖直的线，表示取的最佳的alpha的值对应的线，用虚线表示
plt.axvline(model_lassoCV.alpha_,color = 'black',ls='--')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.show()