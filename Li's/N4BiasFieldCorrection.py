import SimpleITK as sitk

imagePath = '/Users/mac/Documents/simpleITK/T1C_Data.nrrd'
input_image = sitk.ReadImage(imagePath)
#本身没有，通过SimpleITK这个包生成一个mask，数据校正只在mask中进行
mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
input_image = sitk.Cast(input_image, sitk.sitkFloat32)#数据类型转换
corrector = sitk.N4BiasFieldCorrectionImageFilter()#初始化N4偏置场函数
output_image = corrector.Execute(input_image, mask_image)
output_image = sitk.Cast(output_image, sitk.sitkInt16)#根据具体情况选择输出影像
sitk.WriteImage(output_image, '/Users/mac/Documents/simpleITK/N4.nii')
sitk.WriteImage(mask_image, '/Users/mac/Documents/simpleITK/mask.nii')
print('1')