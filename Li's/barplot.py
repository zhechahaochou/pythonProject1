import seaborn as sns
import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt

#用自带的数据测试
tips = pg.read_dataset('tips')
print(tips)
#画柱状图📊
plt.figure()
sns.barplot(data=tips, x='day', y='total_bill')
plt.show()

#优化柱状图
plt.figure()
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', #横坐标根据sex再分类
            order=['Thur','Fri','Sat','Sun'],#调整横坐标顺序
            estimator=np.median,#误差棒默认是均值，这里调为中位数
            palette='Blues_d',#柱子色调（可根据seaborn手册看）
            capsize=.1)#误差棒帽长度，默认为0
plt.show()
#横向柱状图
plt.figure()
sns.barplot(data=tips, x='tip', y='size',orient='h')#orient='h'水平；orient='v'竖直
plt.show()
