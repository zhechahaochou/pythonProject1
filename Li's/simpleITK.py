import SimpleITK as sitk
folderPath = '/Users/mac/Documents/simpleITK/'
# #把文件读到程序里（一系列doc文件）
# reader = sitk.ImageSeriesReader()
# dicom_names = reader.GetGDCMSeriesFileNames(folderPath) #读取路径下的文件名
# reader.SetFileNames(dicom_names)
# image = reader.Execute()
# #保存图像（dicom->nii）
# sitk.WriteImage(image,folderPath+'test.nii.gz')

maskFilePath = '/Users/mac/Documents/simpleITK/T1C_Data.nrrd'
# reader = sitk.ImageFileReader()
# reader.SetFileName(maskFilePath)
# image = reader.Execute()
image = sitk.ReadImage(maskFilePath)
print('1')
#对图像进行处理
image_arr = sitk.GetArrayFromImage(image)#得到的矩阵顺序：z,y,x
size = image.GetSize()
origin = image.GetOrigin() #影像起始点所在坐标 顺序：x,y,z
spacing = image.GetSpacing() #每个点在x,y,z方向上是多少
direction = image.GetDirection()

pixelType = sitk.sitkUInt8
image_new = sitk.Image(size, pixelType)#新建图像
image_arr_new = image_arr[:,:,::-1]#：：-1镜像
image_new = sitk.GetImageFromArray(image_arr_new)
image_new.SetDirection(direction)
image_new.SetSpacing(spacing)
image_new.SetOrigin(origin)
sitk.WriteImage(image_new,folderPath+'test_reverseX.nii')