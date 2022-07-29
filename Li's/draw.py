import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

'''特征权重作图及美化'''
x_values = np.arange(len(index)) #横坐标为筛选出的特征值的个数（15个）———>0，1,，,2，...14
y_values = coef[coef!=0] #特征值
plt.bar(x_values,y_values #画柱状图
        ,color = 'lightblue' #柱子颜色
        ,edgecolor = 'black' #对柱子描边的颜色
        ,alpha = 0.8 #整个柱子的不透明度（包括描的边）
        )
plt.xticks(x_values,index #刚刚x轴的值是0，1...14，现在把他改成对应特征的名字
           ,rotation = 45 #旋转45度
           ,ha = 'right' #Horizontal aline水平对齐
           ,va = 'top' #Vertical aline竖直对齐
           )
plt.xlabel('feature') #设置x轴的名称
plt.ylabel('weight') #设置y轴的名称
plt.show()