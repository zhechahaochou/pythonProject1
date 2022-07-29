import pandas as pd
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
#关闭ssl本地验证
ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv')
print(df.head())
#得到特征之间相关系数的矩阵（主对角线对称）
#method参数默认是计算pearson相关系数（要求正态分布），另可计算kendall(针对分类数据)、spearman（不要求正态分布）
print(df.corr())
#做热度图
plt.figure(figsize=(12,10),dpi=80)
sns.heatmap(df.corr(),xticklabels=df.corr().columns,yticklabels=df.corr()
            ,cmap='RdYlGn',center=0,annot=True)
#装饰
plt.title('Correlogram of mtcars',fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()