# from networks.ResUnet.dim3.unet import UNet
# from networks.ResUnet.dim3.CrossUnet_3D_5 import CrossUnet
# from networks.ResUnet.dim3.FilterTransU_3D import FilterTransU_3D
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.SwinUnter_unofficial import SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.Cotr.Cotr import U_ResTran3D
from networks.UNetpp.unetpp import UNetPlusPlus
from networks.ResUnet.dim3.baseline import BaseUnet # 残差链接的unet LL baseline
from networks.ResUnet.dim3.ReCross5.ResCross_5 import ResCross_5
# -- coding: utf-8 --
import torchvision
from ptflops import get_model_complexity_info

# model = M2SNet()
model = ResCross_5(1,32)
flops, params = get_model_complexity_info(model.cuda(), (1, 96, 96, 96), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)



