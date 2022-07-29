from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import  Counter
from imblearn.over_sampling import SMOTE
#随机生成两类数据，一类10个，一类90个
#X:每个点的位置；y:每个点的值（0，1,）即特征类别
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9,0.1], n_informative=2,
                           n_redundant=0, flip_y=0,
                           n_features=2, n_clusters_per_class=1,
                           n_samples=100, random_state=1)
print(Counter(y))#y（特征）的种类和个数
#画散点图 X[:,0]:取二维数据中第一维的所有数据；hue=y：根据y分类
plt.figure()
sns.scatterplot(X[:,0], X[:,1], hue=y)
plt.show()
#smot数据合成
smo = SMOTE(random_state=42)#括号中表示每次都按这个方式随机，可去掉
X_smo, y_smo = smo.fit_sample(X,y)
print(Counter(y_smo))

plt.figure()
sns.scatterplot(X_smo[:,0], X_smo[:,1], hue=y_smo, palette='Accent')#画新生成的数据，颜色为Accent
sns.scatterplot(X[:,0], X[:,1], hue=y)#画原来的数据
plt.show()