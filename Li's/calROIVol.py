import SimpleITK as sitk
import numpy as np
#mask（roi）文件的路径
maskFilePath = '/Users/mac/Documents/simpleITK/T1C_Data.nrrd'
#将roi文件读到系统中
reader = sitk.ImageFileReader()
reader.SetFileName(maskFilePath)
mask = reader.Execute()

maskArr = sitk.GetArrayFromImage(mask) #图像->数组 #order: z,y,x
counts = np.sum(mask == 1) #一般将roi区域内标为1，其他标为0
spacing = mask.GetSpacing() #获取体素与体素间的距离 #order：x,y,z
unitVol = np.prod(spacing)#product乘法：将长宽高相乘得体积
roiVol = unitVol * counts #单位：立方毫米
print(roiVol)