import six
import os
import numpy as np
from radiomics import featureextractor
import pandas as pd
from sklearn.utils import shuffle

fp_1 = '/Users/mac/Documents/yan/results_1.xlsx'
fp_2 = '/Users/mac/Documents/yan/results_2.xlsx'
fp_3 = '/Users/mac/Documents/yan/results_3.xlsx'
data_1 = pd.read_excel(fp_1)
data_2 = pd.read_excel(fp_2)
data_3 = pd.read_excel(fp_3)
data = pd.concat([data_1, data_2, data_3])
data.to_excel("/Users/mac/Documents/yan/results.xlsx")

