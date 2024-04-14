# from sklearn.model_selection import KFold
from monai.data import CacheDataset
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel


from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    Invertd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd
)

import numpy as np
from collections import OrderedDict
import glob

def data_loader(args):
    root_dir = args.root
    dataset = args.dataset

    print('Start to load data from directory: {}'.format(root_dir))

    if dataset == 'feta':
        out_classes = 8
    elif dataset == 'mmwhs':
        out_classes = 8
    elif dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16
    elif dataset == 'NSCLC':
        out_classes = 2
    elif dataset == 'ABD_Task':
        out_classes = 5
    elif dataset == 'ACDC':
        out_classes = 4
    elif dataset == 'synpase':
        out_classes = 14
    elif dataset == 'SegTHOR':
        out_classes = 5

    if args.mode == 'train':
        train_samples = {}
        valid_samples = {}

        ## Input training data
        train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
        train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        train_samples['images'] = train_img
        train_samples['labels'] = train_label

        ## Input validation data
        valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
        valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        ## Input inference data
        test_img = sorted(glob.glob(os.path.join(root_dir, '*.nii.gz')))
        test_samples['images'] = test_img

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes



def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None


    if dataset == 'feta':  # 每个函数加d的原因在于以字典的形式进行变换   使用字典变换时，必须指明该变换是对image做，还是label做。如，LoadImaged（keys='image'）,表明只加载image
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]), # 向输入图像添加一个长度为 1 的通道维度。
                Orientationd(keys=["image", "label"], axcodes="RAS"), # 方向变换
                ScaleIntensityRanged(  #ScaleIntensityRanged可以指定把哪些范围值缩放到那个区间      这个 a_min=0, a_max=1000 参数需要去查阅mmwhs论文看一下范围是多少  自己统计的是100-1000
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    # if dataset == 'BTCV':
    #     train_transforms = Compose(
    #         [
    #             LoadImaged(keys=["image", "label"]),
    #             AddChanneld(keys=["image", "label"]),
    #             # Spacingd(keys=["image", "label"], pixdim=(
    #             #     0.8, 1.0, 1.0), mode=("bilinear", "nearest")),  # 重采样  第一个2.0是切片数量扩充到多少倍
    #             Orientationd(keys=["image", "label"], axcodes="RAS"),
    #             ScaleIntensityRanged(
    #                 keys=["image"], a_min=-100, a_max=400,
    #                 b_min=0.0, b_max=1.0, clip=True,
    #             ),
    #             CropForegroundd(keys=["image", "label"], source_key="image"),
    #             RandCropByPosNegLabeld(
    #                 keys=["image", "label"],
    #                 label_key="label",
    #                 spatial_size=(96, 96, 96),
    #                 pos=1,
    #                 neg=1,
    #                 num_samples=crop_samples,
    #                 image_key="image",
    #                 image_threshold=0,
    #             ),
    #             # SpatialPadd(  # 防止裁剪后的图像大小小于96,96,96
    #             #     keys=["image", "label"],
    #             # spatial_size = (96, 96, 96), method = 'symmetric', mode = 'constant'
    #             # ),
    #             # RandShiftIntensityd(
    #             #     keys=["image"],
    #             #     offsets=0.10,
    #             #     prob=0.50,
    #             # ),
    #             # RandAffined(
    #             #     keys=['image', 'label'],
    #             #     mode=('bilinear', 'nearest'),
    #             #     prob=1.0, spatial_size=(96, 96, 96),
    #             #     rotate_range=(0, 0, np.pi / 30),
    #             #     scale_range=(0.1, 0.1, 0.1)),
    #             ToTensord(keys=["image", "label"]),
    #         ]
    #     )
    #
    #     val_transforms = Compose(
    #         [
    #             LoadImaged(keys=["image", "label"]),
    #             AddChanneld(keys=["image", "label"]),
    #             # Spacingd(keys=["image", "label"], pixdim=(
    #             #     0.8, 1.0, 1.0), mode=("bilinear", "nearest")),
    #             Orientationd(keys=["image", "label"], axcodes="RAS"),
    #             ScaleIntensityRanged(
    #                 keys=["image"], a_min=-100, a_max=400,
    #                 b_min=0.0, b_max=1.0, clip=True,
    #             ),
    #             CropForegroundd(keys=["image", "label"], source_key="image"),
    #             ToTensord(keys=["image", "label"]),
    #         ]
    #     )
    #
    #     test_transforms = Compose(
    #         [
    #             LoadImaged(keys=["image"]),
    #             AddChanneld(keys=["image"]),
    #             # Spacingd(keys=["image", "label"], pixdim=(
    #             #     0.8, 1.0, 1.0), mode=("bilinear", "nearest")),
    #             ScaleIntensityRanged(
    #                 keys=["image"], a_min=-100, a_max=400,
    #                 b_min=0.0, b_max=1.0, clip=True,
    #             ),
    #             CropForegroundd(keys=["image"], source_key="image"),
    #             ToTensord(keys=["image"]),
    #         ]
    #     )


    if dataset == 'SegTHOR':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-310,
                    a_max=400,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),

                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(keys=["image"], a_min=-310, a_max=400, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

    if dataset == 'ACDC':  # 每个函数加d的原因在于以字典的形式进行变换   使用字典变换时，必须指明该变换是对image做，还是label做。如，LoadImaged（keys='image'）,表明只加载image
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]), # 向输入图像添加一个长度为 1 的通道维度。
                Spacingd(keys=["image", "label"], pixdim=(
                    2.0, 2.0, 0.4), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"), # 方向变换
                ScaleIntensityRanged(  #ScaleIntensityRanged可以指定把哪些范围值缩放到那个区间      这个 a_min=0, a_max=1000 参数需要去查阅mmwhs论文看一下范围是多少  自己统计的是100-1000
                    keys=["image"], a_min=0, a_max=700,
                    b_min=0.0, b_max=1.0, clip=True,  # 布尔值。设置为True, 才会把[阈值]之外的值都设置为0.通常为True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(  # 按照阴性阳性比裁剪成几个子图
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.30,
                ),
                # RandAffined(
                #     keys=['image', 'label'],
                #     mode=('bilinear', 'nearest'),
                #     prob=0.5, spatial_size=(96, 96, 96),
                #     rotate_range=(0, 0, np.pi / 15),
                #     scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=700,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=700,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    # 对于混合的CT MRI数据格式来讲 先试试不重采样看能不能运行
    if dataset == 'mmwhs':  # 每个函数加d的原因在于以字典的形式进行变换   使用字典变换时，必须指明该变换是对image做，还是label做。如，LoadImaged（keys='image'）,表明只加载image
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]), # 向输入图像添加一个长度为 1 的通道维度。
                # Spacingd(keys=["image", "label"], pixdim=(
                #     0.8, 1.0, 1.0), mode=("bilinear", "nearest")),  # 重采样
                Orientationd(keys=["image", "label"], axcodes="RAS"), # 方向变换
                ScaleIntensityRanged(  #ScaleIntensityRanged可以指定把哪些范围值缩放到那个区间      这个 a_min=0, a_max=1000 参数需要去查阅mmwhs论文看一下范围是多少  自己统计的是100-1000
                    keys=["image"], a_min=0, a_max=700,
                    b_min=0.0, b_max=1.0, clip=True,  # 布尔值。设置为True, 才会把[阈值]之外的值都设置为0.通常为True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(  # 按照阴性阳性比裁剪成几个子图
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.30,
                ),
                # RandAffined(
                #     keys=['image', 'label'],
                #     mode=('bilinear', 'nearest'),
                #     prob=0.5, spatial_size=(96, 96, 96),
                #     rotate_range=(0, 0, np.pi / 15),
                #     scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     2.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=700,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     2.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=700,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'flare':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear")),
                # ResizeWithPadOrCropd(keys=["image"], spatial_size=(168,168,128), mode=("constant")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'amos':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                # RandAffined(
                #     keys=['image', 'label'],
                #     mode=('bilinear', 'nearest'),
                #     prob=1.0, spatial_size=(96, 96, 96),
                #     rotate_range=(0, 0, np.pi / 30),
                #     scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'NSCLC':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-500, a_max=500,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                # RandShiftIntensityd(
                #     keys=["image"],
                #     offsets=0.10,
                #     prob=0.50,
                # ),
                # RandAffined(
                #     keys=['image', 'label'],
                #     mode=('bilinear', 'nearest'),
                #     prob=1.0, spatial_size=(96, 96, 96),
                #     rotate_range=(0, 0, np.pi / 30),
                #     scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-500, a_max=500,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-500, a_max=500,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'ABD_Task':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    2.0, 1.0, 1.0), mode=("bilinear", "nearest")),  # 重采样  第一个2.0是切片数量扩充到多少倍
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-400, a_max=400,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                # SpatialPadd(  # 防止裁剪后的图像大小小于96,96,96
                #     keys=["image", "label"],
                # spatial_size = (96, 96, 96), method = 'symmetric', mode = 'constant'
                # ),
                # RandShiftIntensityd(
                #     keys=["image"],
                #     offsets=0.10,
                #     prob=0.50,
                # ),
                # RandAffined(
                #     keys=['image', 'label'],
                #     mode=('bilinear', 'nearest'),
                #     prob=1.0, spatial_size=(96, 96, 96),
                #     rotate_range=(0, 0, np.pi / 30),
                #     scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    2.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-400, a_max=400,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    2.0, 1.0, 1.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-400, a_max=400,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'synpase':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image" ]),
                Orientationd(keys=["image" ], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image" ], source_key="image"),
            ]
        )


    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms


def infer_post_transforms(args, test_transforms, out_classes):
    # pre_transforms =  Compose(
    #     [
    #         LoadImaged(keys="img"),
    #         EnsureChannelFirstd(keys="img"),
    #         Orientationd(keys="img", axcodes="RAS"),
    #         Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
    #         ScaleIntensityd(keys="img"),
    #     ]
    # )

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        # Invertd这个不知道为什么不行
        # Invertd(
        #     keys="pred",  # invert the `pred` data field, also support multiple fields
        #     transform=test_transforms,
        #     orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
        #     # then invert `pred` based on this information. we can use same info
        #     # for multiple fields, also support different orig_keys for different fields
        #     meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
        #     orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
        #     # for example, may need the `affine` to invert `Spacingd` transform,
        #     # multiple fields can use the same meta data to invert
        #     meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
        #     # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
        #     # otherwise, no need this arg during inverting
        #     nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
        #     # to ensure a smooth output, then execute `AsDiscreted` transform
        #     to_tensor=True,  # convert to PyTorch Tensor after inverting
        # ),
        # Invertd(
        #     keys="pred",  # invert the `pred` data field, also support multiple fields
        #     transform=test_transforms,
        #     orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
        #     # then invert `pred` based on this information. we can use same info
        #     # for multiple fields, also support different orig_keys for different fields
        #     nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
        #     # to ensure a smooth output, then execute `AsDiscreted` transform
        #     to_tensor=True,  # convert to PyTorch Tensor after inverting
        # ),
        AsDiscreted(keys="pred", argmax=True), # to ensure a smooth output, then execute `AsDiscreted` transform
        KeepLargestConnectedComponentd(keys='pred', applied_labels=range(1,out_classes)),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms



