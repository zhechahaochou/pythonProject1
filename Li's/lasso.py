import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
fp1 = '/Users/mac/Documents/clinical.xlsx'
data_1 = pd.read_excel(fp1)
rows_1,_ = data_1.shape
data = shuffle(data_1)
data = data.fillna(0)

X = data[data.columns[1:]]
y = data['Label']
colNames= X.columns
X = StandardScaler().fit_transform(X) #标准化尺度
X = pd.DataFrame(X)
X.columns = colNames

alphas = np.logspace(-3,1,50)
model_lassoCV = LassoCV(alphas = alphas,cv = 10,max_iter=10000).fit(X,y)

# print(model_lassoCV.alpha_) #最佳alpha值
coef = pd.Series(model_lassoCV.coef_, index=X.columns)
# print(ceof)
#Lasso选出了22个变量（特征值） 排除筛选掉了其他26个变量
# print('Lasso picked' + str(sum(ceof != 0)) + 'variables and eliminated the other' + str(sum(ceof == 0)))

index = coef[coef !=0].index #筛选出来不为零特征值的名称
X = X[index]
print(X.head())
# print(coef[coef != 0])