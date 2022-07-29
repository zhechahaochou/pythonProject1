import pingouin as pg
import pandas as pd
import numpy as np
import os
# #用自带的数据试验pingouin
# data = pg.read_dataset('icc') #自带数据
# #组内相关系数
# icc = pg.intraclass_corr(data = data, #数据来源
#                          targets='Wine', #病的种类
#                          raters='Judge', #医生
#                          ratings='Scores') #打分
# print(icc)

#用自己编的数据测试
folderPath = '/Users/mac/Documents/icc/'
data_1 = pd.read_excel(os.path.join(folderPath,'reader_1.xlsx'))
data_2 = pd.read_excel(os.path.join(folderPath,'reader_2.xlsx'))
#0：在最前面新建一列，取名为reader；建立一堆1（医生编号），个数是data_1的行数；建立一堆2，个数是data_2的行数
data_1.insert(0,'reader',np.ones(data_1.shape[0]))
data_2.insert(0,'reader',np.ones(data_2.shape[0])*2)
data_1.insert(0,'target',range(data_1.shape[0]))
data_2.insert(0,'target',range(data_2.shape[0]))
data = pd.concat([data_1,data_2])#合并两个表
print(data)
#icc
icc = pg.intraclass_corr(data=data, targets='target', raters='reader', ratings='featureA')
print(icc)