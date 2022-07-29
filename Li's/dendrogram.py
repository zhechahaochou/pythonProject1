import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
filePath = '/Users/mac/Documents/clinical.xlsx'
data = pd.read_excel(filePath)
data = shuffle(data)
data = data.fillna(0)
data_1 = data.loc[data['Label'].isin([0])]
data_2 = data.loc[data['Label'].isin([1])]

X = data[data.columns[1:]]
y = data['Label']
colNames = X.columns
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames


#lasso
alphas = np.logspace(-3, 1, 50)
model_lassoCV = LassoCV(alphas = alphas, cv = 10, max_iter=100000).fit(X,y)
coef = pd.Series(model_lassoCV.coef_, index=X.columns)
print(model_lassoCV.alpha_)
print('Lasso picked '+ str(sum(coef!=0))+ ' variables and eliminated the other '+ str(sum(coef==0)) + ' variables')
print(coef[coef != 0])
X = X[coef[coef !=0].index]
print(X.head())
#dendrogram聚类树状图
plt.figure(figsize=(5,5), dpi=80)
plt.title('Radiomics', fontsize=22)
#shc.linkage（需要的特征，将这些特征连起来的方法）
#labels=X.columns 相当于病的种类
#color_threshold对应的是纵坐标的值，值以上为1个颜色，以下为不同颜色
dend = shc.dendrogram(shc.linkage(X[:].T, method='ward'),
                      labels=X.columns, color_threshold=20)#color_threshold对应的是纵坐标的值，值以上为1个颜色，以下为不同颜色
plt.xticks(fontsize=12, rotation=60, ha='right')#x轴坐标旋转60度，右对齐
plt.show()