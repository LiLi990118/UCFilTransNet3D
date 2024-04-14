import numpy as np
import nibabel as nib
import os

def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
    '''
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param class_num: total number of categories
    :return:
    '''
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float64)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float64)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float64)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice
#
# intersection = torch.sum(gt_tensor * pred_tensor)
# union = torch.sum(gt_tensor) + torch.sum(pred_tensor)
# dice_score = (2.0 * intersection) / (union + 1e-7)  # 添加一个小的常数以避免除以零

def evaluate_demo(prediction_nii_files, target_nii_files):
    '''
    This is a demo for calculating the mean dice of all subjects.
    :param prediction_nii_files: a list which contains the .nii file paths of predicted segmentation
    :param target_nii_files: a list which contains the .nii file paths of ground truth mask
    :return:
    '''
    dscs = []
    for prediction_nii_file, target_nii_file in zip(prediction_nii_files, target_nii_files):
        prediction_nii = nib.load(prediction_nii_file)
        prediction = prediction_nii.get_fdata()
        target_nii = nib.load(target_nii_file)
        target = target_nii.get_fdata()
        # target[target>0]=1 # 仅限于二分类
        dsc = cal_subject_level_dice(prediction, target, class_num=5)
        dscs.append(dsc)
    return np.mean(dscs)

pred_nii_dir = r'E:\ZQH\Subtask1\PpredsTs'
label_nii_dir = r'E:\ZQH\Subtask1\labelsTs'

label_nii_list = []
pred_nii_list = []

dice_list = []

#
for name in os.listdir(label_nii_dir):
    label_nii_list.append(name) # mr_train_1003_label.nii.gz

for name in os.listdir(pred_nii_dir):
    pred_nii_list.append(name+'_seg.nii.gz') # mr_train_1003_label.nii.gz

for i in range(len(pred_nii_list)):
    pred = pred_nii_list[i]
    pred_nii = os.path.join(pred_nii_dir,pred.split('.')[0].replace('_seg',''))
    pred_nii_path = os.path.join(pred_nii,pred)
    # print(pred_nii_path)
    label_nii_path = os.path.join(label_nii_dir,label_nii_list[i])
    print(evaluate_demo([pred_nii_path],[label_nii_path]))#前一个地址是你预测的三维图像的地址，后一个是标签地址)
    dice_list.append(evaluate_demo([pred_nii_path],[label_nii_path]))

print("avg dice:",np.mean(dice_list))



# pred_nii_path = r'C:\Users\admin\Desktop\NSCLC\NSCLS_3DSeg\PredsTs\s300_image\s300_image_seg.nii.gz'
# label_nii_path = r'C:\Users\admin\Desktop\NSCLC\NSCLS_3DSeg\LabelsTs\s300_mask.nii.gz'
# print(evaluate_demo([pred_nii_path],[label_nii_path]))