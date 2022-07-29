from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
'''标准化'''
#z = (x-u)/s
#异常值一般对结果影响不大
# data = [[10,-20],[0.3,999],[-1,12],[0.1,21]]
# res = StandardScaler().fit_transform(data)
# print(res)
'''归一化'''
#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) )
#异常值可能对结果有重要影响
data = [[10,-20],[0.3,999],[-1,12],[0.1,21]]
res = MinMaxScaler().fit_transform(data)
print(res)
