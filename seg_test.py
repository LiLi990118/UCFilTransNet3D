import numpy as np
from monai.metrics import DiceMetric
from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete

from networks.ResUnet.dim3.unet import UNet  # 原始的unet
from networks.SilmUnetr.SlimUNETR import SlimUNETR
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

# from networks.Unet_3d import UNet  # 最原始的unet
from networks.ResUnet.dim3.baseline import BaseUnet
from networks.ResUnet.dim3.CrossUnet_3D_831 import CrossUnet
from networks.Cotr.Cotr import U_ResTran3D
from networks.ResUnet.dim3.ReCross5.ResCross_5 import ResCross_5
from networks.ResUnet.dim3.Cross_bo_Cat.Cross_bo_cat import Cross_bo_cat
import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms

import os
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default=r'D:\ZQH\3D_Seg_Dataset\Synpase_new\ImagesTs', help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default=r'D:\ZQH\3D_Seg_Dataset\synpase_Model_Weights\synpase_Task_SegResNet_predsTs', help='Output folder for both tensorboard and the best ResUnet')
parser.add_argument('--dataset', type=str, default='synpase', help='Datasets: {feta, flare, amos, mmwhs,NSCLC,ABD_Task}, Fyi: You can add your dataset here')

## Input ResUnet & training hyperparameters
parser.add_argument('--network', type=str, default='SegResNet', help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET,3DUnet,CrossUnet,CoTr}')
parser.add_argument('--trained_weights', default=r'D:\ZQH\3D_Seg_Dataset\synpase_Model_Weights\SegResNet_synpase\best_metric_model.pth', help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=2, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.8, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.5, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name} for image_name in zip(test_samples['images'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

## Inference Pytorch Data Loader and Caching
test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

# label and pred
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

import numpy as np
import nibabel as nib





## Load Networks
device = torch.device("cuda:0")
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
elif args.network == '3DUnet':
    model = UNet(in_ch=1, base_ch=16,num_classes=out_classes).to(device)
elif args.network == 'CrossUnet':
    model = Cross_bo_cat(1, 5).to(device)
elif args.network == 'CoTr':
    model = U_ResTran3D(num_classes=out_classes).to(device)

elif args.network == 'SegResNet':
    model = SegResNet(blocks_down=[1, 2, 3, 4],
    blocks_up=[1, 1, 1],
    init_filters=32,
    in_channels=1,
    out_channels=out_classes,
    dropout_prob=0.1,).to(device)
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


model.load_state_dict(torch.load(args.trained_weights))
model.eval()
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)
        roi_size = (96, 96, 96)
        test_data['pred'] = sliding_window_inference(
            images, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        # print(test_data.shape)