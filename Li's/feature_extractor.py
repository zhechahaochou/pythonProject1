import radiomics
from radiomics import featureextractor
import pandas as pd


#原图  /Users/mac/Documents/feature_extractor /Users/mac/Documents/feature_extractor/P000128771_t2.nii
#ROI  /Users/mac/Documents/feature_extractor/P000128771_T2\(1\).nii
img_path = '/Users/mac/Documents/feature_extractor/P000128771_t2.nii'
roi_path = '/Users/mac/Documents/feature_extractor/P000128771_t2_roi.nii'
extractor = featureextractor.RadiomicsFeatureExtractor()
df = pd.DataFrame()
featureVector = extractor.execute(img_path,roi_path)
df_add = pd.DataFrame.from_dict(featureVector.values()).T
df_add.columns = featureVector.keys()
df = pd.concat([df,df_add])
df.to_excel('/Users/mac/Documents/feature_extractor/results.xlsx')