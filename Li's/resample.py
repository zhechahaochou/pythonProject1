import SimpleITK as sitk
import numpy as np

imagePath = '/Users/mac/Documents/simpleITK/T1C_Data.nrrd'
image = sitk.ReadImage(imagePath)
resample = sitk.ResampleImageFilter()#初始化
resample.SetInterpolator(sitk.sitkLinear)#设置插值方法：线性插值
resample.SetOutputDirection(image.GetDirection())#输出影像方向
resample.SetOutputOrigin(image.GetOrigin())#设置原点

newSpacing = [0.5, 0.5, 0.5]
newSpacing = np.array(newSpacing, float)
newSize = image.GetSize() / newSpacing * image.GetSpacing()#新大小=原大小*原间隔/新间隔
newSize = newSize.astype(int)#格式转换
resample.SetSize(newSize.tolist())#格式转换
resample.SetOutputSpacing(newSpacing)
newimage = resample.Execute(image)
sitk.WriteImage(newimage, '/Users/mac/Documents/simpleITK/resample.nii')
print(image.GetSize())
print(resample.GetSize())