#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""

from monai.utils import set_determinism
from monai.transforms import AsDiscrete

from networks.MedNext import MedNeXt
from networks.ResUnet.dim3.unet import UNet  # 原始的unet
# from networks.ResUnet.dim3.baseline import BaseUnet # 残差链接的unet LL baseline
# # from networks.ResUnet.dim3.CrossUnet_3D_5 import CrossUnet
# from networks.ResUnet.dim3.ReCross5.ResCross_5 import ResCross_5
# # from networks.ResUnet.dim3.Cross_bo_Cat.Cross_bo_cat import  Cross_bo_cat
# # from networks.ResUnet.dim3.FilterTransU_3D import FilterTransU_3D
# from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
# from networks.SwinUnter_unofficial import SwinUNETR
from networks.UXNet_3D.network_backbone import UXNET
from networks.Unest.unest_small_patch_4 import UNesT
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.Cotr.Cotr import U_ResTran3D
from networks.dca_3D.effiunet_plus_DCA import effiunet_plus_dca_unet
from networks.ResUnet.dim3.CrossUnet_3D_831 import CrossUnet
from networks.SilmUnetr.SlimUNETR import SlimUNETR
# from networks.UNetpp.unetpp import UNetPlusPlus

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch,load_decathlon_datalist

import torch
from tensorboardX  import SummaryWriter
from load_datasets_transforms import data_transforms, data_loader

import os
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
## Input data hyperparameters  #E:\ZQH\Subtask1   E:\ZQH\3D_Seg_mmwhs
parser.add_argument('--root', type=str, default=r'D:\ZQH\3D_Seg_Dataset\MMWHS', help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default=r'D:\ZQH\3D_Seg_Dataset\mmwhs_Model_Weights\UNesT_mmwhs',help='Output folder for both tensorboard and the best ResUnet')
parser.add_argument('--dataset', type=str, default='mmwhs', help='Datasets: {feta, flare, amos, mmwhs, NSCLC,ABD_Task,ACDC,BTCV,SegTHOR,synpase}, Fyi: You can add your dataset here')

## Input ResUnet & training hyperparameters
parser.add_argument('--network', type=str, default='UNesT', help= 'Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, CrossUnet,Unetpp'
                                                                  '3DUXNET,FilterTrans3D,3DUnet,CoTr,eff_plus_cfa,slim_unetr,UNesT,MedNext}' )
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='2', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='3', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=20000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=200, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1.0, help='Cache rate to cache your dataset into GPUs')
# If the error "Out of GPU memory" is popped out, please reduce the number of crop_sample or cache_rate
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))


# def data_loader(args):
#     root_dir = args.root
#     dataset = args.dataset
#
#     print('Start to load data from directory: {}'.format(root_dir))
#
#     if dataset == 'feta':
#         out_classes = 8
#     elif dataset == 'mmwhs':
#         out_classes = 8
#     elif dataset == 'flare':
#         out_classes = 5
#     elif dataset == 'amos':
#         out_classes = 16
#
#     return out_classes
#
# out_classes =  data_loader(args)

# 这里加载数据格式为(nii.gz)
train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

# split_JSON = r"E:\ZQH\3DUX-Net-main\AMOS\dataset_AMOS.json"
split_JSON = r"E:\ZQH\3DUX-Net-main\mmwhs-Data\dataset_MMWHS.json"

# train_files = load_decathlon_datalist(split_JSON, True, "training")
# val_files = load_decathlon_datalist(split_JSON, True, "val")

set_determinism(seed=3471)

train_transforms, val_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


## Load Networks
device = torch.device("cuda:0")
# device = torch.device("cpu")
if args.network == '3DUXNET':
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
elif args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)

elif args.network == 'SegResNet':
    model = SegResNet(blocks_down=[1, 2, 3, 4],
    blocks_up=[1, 1, 1],
    init_filters=32,
    in_channels=1,
    out_channels=out_classes,
    dropout_prob=0.1,).to(device)

elif args.network == 'nnFormer':
    model = nnFormer(input_channels=1, num_classes=out_classes).to(device)

elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=out_classes,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
elif args.network == 'TransBTS':
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)
elif args.network == 'FilterTrans3D':
    model = FilterTransU_3D(1, 16)
    model = model.to(device)
elif args.network == '3DUnet':
    model = UNet(in_ch=1, base_ch=16,num_classes=out_classes).to(device)
elif args.network == 'CoTr':
    model = U_ResTran3D(num_classes=out_classes).to(device)
elif args.network == 'CrossUnet':
    model = CrossUnet(1, 8).to(device)
elif args.network == 'Unetpp':
    model = UNetPlusPlus(1, 32).to(device)
elif args.network == 'eff_plus_cfa':
    model_config = [[1, 3, 1, 1, 24, 24, 0, 0],
                    [1, 3, 2, 4, 24, 48, 0, 0],
                    [1, 3, 2, 4, 48, 64, 0, 0],
                    [1, 3, 2, 4, 64, 128, 1, 0.25],
                    [1, 3, 2, 6, 128, 160, 1, 0.25],
                    [1, 3, 1, 6, 160, 256, 1, 0.25]
                    ]
    model = effiunet_plus_dca_unet(n_channels=1, n_classes=8, model_config=model_config).to(device)
elif args.network == 'slim_unetr':
    model = SlimUNETR(
        in_channels=1,
        out_channels=out_classes,
        embed_dim=96,
        embedding_dim=27,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ).to(device)
elif args.network == 'UNesT':
    model = UNesT(in_channels=1,
                        out_channels=out_classes,
                        patch_size=4,
                        depths = [2, 2, 8],
                        num_heads=[2, 4, 8],
                        embed_dim=[64, 128, 256]).to(device)
elif args.network == 'MedNext':
    num_block = [2, 1, 1, 1, 1, 1, 1, 1, 2]
    scale = [2, 1, 1, 1, 1, 1, 1, 1, 1]
    model = MedNeXt(1, 32, 3, num_block, scale, out_classes).to(device)

print('Chosen Network Architecture: {}'.format(args.network))

if args.pretrain == 'True':
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights))

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-5)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.999), weight_decay=5e-4)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)


root_dir = os.path.join(args.output)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)
    
t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            # val_outputs = ResUnet(val_inputs)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model,overlap=0.6) # 概率图
            # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)  # val_labels_convert 此时已经转换成了2通道
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('Validation_Segmentation_Loss', mean_dice_val, global_step)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    # model_feat.eval()
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        # with torch.no_grad():
        #     g_feat, dense_feat = model_feat(x)
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            if global_step == max_iterations:
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "last_metric_model.pth")
                )
                # scheduler.step(dice_val)

        writer.add_scalar('Training_Segmentation_Loss', loss.data.cpu().numpy(), global_step)
        global_step += 1


    return global_step, dice_val_best, global_step_best


max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)  # AsDiscrete argmax=True将概率图转换成分割图   to_onehot=out_classes做onehot编码
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step <= max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )





