import six
import os
import numpy as np
from radiomics import featureextractor
import pandas as pd
import datetime

dataDir = '/Users/mac/Documents/yan/'
folderList = os.listdir(dataDir)  # 取路径下文件
folderList.remove('.DS_Store')  # mac
folderList.sort()
folderList_1 = folderList[0:30:1]
folderList_2 = folderList[30:60:1]
folderList_3 = folderList[60:95:1]


# folderList = ['ms002']
extractor = featureextractor.RadiomicsFeatureExtractor('/Users/mac/PycharmProjects/pythonProject1/CognitiveImpairment'
                                                       '/Params_zhang.yaml')
df = pd.DataFrame()

for folder in folderList_3:
    imageName = dataDir + folder + '/coT1.nii'
    maskName = dataDir + folder + '/CLs.nii.gz'
    featureVector = extractor.execute(imageName, maskName)
    df_add = pd.DataFrame.from_dict(featureVector.values()).T
    df_add.columns = featureVector.keys()
    df = pd.concat([df,df_add])
df.to_excel(dataDir + 'results_3.xlsx')



