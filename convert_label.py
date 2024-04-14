import nibabel as nib
import os
import glob
import shutil
# 读取NIfTI文件





path1 = r'C:\Users\admin\Desktop\NSCLC\NSCLS_3DSeg'
file_path = r'C:\Users\admin\Desktop\NSCLC\NSCLS_3DSeg\ImagesVal'

# 重命名
# for i in os.listdir(file_path):
#     file_dir = os.path.join(file_path,i)
#     nii_name = glob.glob(os.path.join(file_dir,'*nii.gz'))[0].split('\\')[-1]
#     print(i+'_'+nii_name)
#     # nii_data = nib.load(file_dir+'//'+nii_name)
#     # image_data = nii_data.get_fdata()
#     # # 将大于0的值改为1
#     # image_data[image_data > 0] = 1
#     # # 创建修改后的NIfTI对象
#     # modified_nii = nib.Nifti1Image(image_data, nii_data.affine, nii_data.header)
#     # nib.save(modified_nii,file_dir+'//'+i+'_'+nii_name)
#     os.rename(file_dir+'//'+nii_name,file_dir+'//'+i+'_'+nii_name)

# 将标签大于0的像素转换成1
# for i in os.listdir(file_path)[:]:
#     file_dir = os.path.join(file_path,i)
#     nii_name = glob.glob(os.path.join(file_dir,'*nii.gz'))[0].split('\\')[-1]
#     print(i+'_'+nii_name)
#     nii_data = nib.load(file_dir+'//'+nii_name)
#     image_data = nii_data.get_fdata()
#     # 将大于0的值改为1
#     image_data[image_data > 0] = 1
#     # 创建修改后的NIfTI对象
#     modified_nii = nib.Nifti1Image(image_data, nii_data.affine, nii_data.header)
#     nib.save(modified_nii,file_dir+'//'+i+'_'+nii_name)
#     os.remove(file_dir+'//'+nii_name)

# 把nii文件移动
# path2 = r'C:\Users\admin\Desktop\NSCLC\NSCLS_3DSeg\ImagesVal'
# for i in os.listdir(path2):
#     file_dir = os.path.join(path2,i)
#     nii_name = glob.glob(os.path.join(file_dir,'*nii.gz'))[0].split('\\')[-1]
#     # print(i+'_'+nii_name)
#     print(file_dir + '//' + nii_name)
#     print(path2 + '//'  + nii_name)
#     shutil.move(file_dir + '//' + nii_name, path2 + '//'  + nii_name)