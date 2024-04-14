# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ResUnet.dim3.unet_utils import inconv, down_block, up_block
from networks.ResUnet.dim3.utils import get_block, get_norm
import pdb
from einops import rearrange
import mpmath
import numpy as np
from networks.ResUnet.dim3.FilterTrans import PatchEmbeddingBlock
from typing import Tuple, Union
from monai.networks.layers import get_act_layer
from monai.utils import look_up_option
from monai.utils import optional_import
import math
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_DROPOUT_MODE = {"vit", "swin"}
###########################################################################
class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        act: Union[Tuple, str] = "GELU",
        dropout_mode="vit",
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: faction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = get_act_layer(act)
        self.drop1 = nn.Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim,h,w,heads=4,  attn_drop=0.):
        super(Attention, self).__init__()
        self.num_attention_heads = heads

        self.dim_head = int(dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.dim_head
        self.complex_weight = nn.Parameter(torch.randn(h,h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)

        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        # self.position_embeddings = nn.Parameter(torch.zeros(1, heads, h*h,h*h))

    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)

    def forward(self, q, x):
        B, N, C = x.shape  # low-dim feature shape
        BH, NH, CH = q.shape  # high-dim feature shape
        zH = aH = bH = int(mpmath.cbrt(mpmath.mpf(NH)))




        # 将输入数据转换为复数张量
        # input_data_complex = torch.view_as_complex(input_data)
        #
        # # 执行傅里叶变换
        # fft_output = fft.fftn(input_data_complex, dim=(-3, -2, -1))


        q = q.view(BH, zH,aH, bH, CH)
        q = q.to(torch.float32)
        # 亮点就是噪声点

        q = torch.fft.rfft2(q, dim=(1, 2,3), norm='ortho')
        weight_q = torch.view_as_complex(self.complex_weight)
        q = q * weight_q
        q = torch.fft.irfft2(q, s=(zH,aH, bH), dim=(1, 2,3), norm='ortho')

        z = a = b = int(mpmath.cbrt(mpmath.mpf(N)))
        #a = b = int(math.sqrt(N))
        x = x.view(B, z,a, b, C)
        x = x.permute(0,4,1,2,3)
        x = F.interpolate(x,size=(zH,aH,bH))
        x = x.permute(0,2,3,4,1)
        x = x.to(torch.float32)
        # 亮点就是噪声点
        x = torch.fft.rfft2(x, dim=(1, 2,3), norm='ortho')
        weight_x = torch.view_as_complex(self.complex_weight)
        x = x * weight_x
        x = torch.fft.irfft2(x, s=(zH,aH, bH), dim=(1, 2,3), norm='ortho')

        mixed_query_layer = q
        mixed_key_layer = x
        mixed_value_layer = x
        mixed_query_layer = rearrange(mixed_query_layer, 'b z h w (dim_head heads) -> b heads (z h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads,
                      z=zH,h=aH, w=bH)
        mixed_key_layer, mixed_value_layer = map(lambda t: rearrange(t, 'b z h w (dim_head heads) -> b heads (z h w) dim_head', dim_head=self.dim_head, heads=self.num_attention_heads, z=zH,h=aH, w=bH), (mixed_key_layer, mixed_value_layer))

        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_scores = torch.einsum('bhid,bhjd->bhij', mixed_query_layer, mixed_key_layer)
        # attention_scores = attention_scores+self.position_embeddings

        # relative_position_bias = self.relative_position_encoding(aH, bH)
        # attention_scores += relative_position_bias

        # attention_scores = mixed_query_layer*mixed_key_layer
        attention_scores = attention_scores / math.sqrt(self.dim_head)
        # attention_probs = self.softmax(attention_scores)
        attention_probs = self.sigmoid(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.einsum('bhij,bhjd->bhid', attention_probs, mixed_value_layer)
        #context_layer = attention_probs*mixed_value_layer


        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = rearrange(context_layer, 'b heads (z h w) dim_head -> b z h w (dim_head heads)', dim_head=self.dim_head, heads=self.num_attention_heads,
                      z=zH,h=aH, w=bH)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output = attention_output.reshape(BH, NH, CH)
        return attention_output
# Unet Transformer building block
class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, num_heads: int, img_size:int,dropout_rate: float = 0.0, qkv_bias: bool = False
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: apply bias term for the qkv linear layer

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        # self.attn = Attention(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.attn = Attention(hidden_size, h=img_size, w=img_size // 2 + 1, heads=num_heads, attn_drop=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        #self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

        #self.mlp = nn.Conv3d(hidden_size, mlp_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x1,x2):
        residue = x1
        # x2(b,28*28,dim)

        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        out = self.attn(x1, x2)
        out = out + residue
        #x = x + self.attn(self.norm1(x))
        x = out + self.mlp(self.norm2(out))
        return x
class FilterTrans(nn.Module):
    def __init__(self, x1_ch,x2_ch, x1_img_size,x2_img_size,dim = 384,heads=4, attn_drop=0.,rel_pos=True,patch_size = (3, 3, 3)):
        super().__init__()
        self.x1_img_size = x1_img_size
        self.patch = 3
        self.hidden_size = dim
        self.embedding1 = PatchEmbeddingBlock(
            in_channels=x1_ch,
            img_size=x1_img_size,
            patch_size=patch_size,
            hidden_size=dim,
            num_heads=heads,
            pos_embed="perceptron",
            dropout_rate=0.0,
            spatial_dims=3,
        )
        self.embedding2 = PatchEmbeddingBlock(
            in_channels=x2_ch,
            img_size=x2_img_size,
            patch_size=patch_size,
            hidden_size=dim,
            num_heads=heads,
            pos_embed="perceptron",
            dropout_rate=0.0,
            spatial_dims=3,
        )
        self.trans = TransformerBlock(hidden_size=dim, mlp_dim=1536, num_heads=heads, img_size= x1_img_size[0]//self.patch,dropout_rate=0.0, qkv_bias=False)

        self.patch_dim = int(x1_ch * np.prod(patch_size))
        self.linear = nn.Linear(384, 378)
        self.out = nn.Conv3d(14, x1_ch,kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, x1, x2):
        # x2: low-dim feature, x1: high-dim feature
        em1 = self.embedding1(x1)

        # x1(b,56*56,dim)

        em2 = self.embedding2(x2)
        #(1,512,384)
        t = self.trans(em1,em2)
        x = self.linear(t)
        b,HWD,_ = x.shape
        h = w = d = int(mpmath.cbrt(mpmath.mpf(HWD)))
        x = rearrange(x,'b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)', p1=self.patch,p2=self.patch,p3=self.patch,d=d,h=h,w=w)
        x = self.out(x)

        return x
class FilterTransU_3D(nn.Module):
    def __init__(self, in_ch, base_ch, scale=[2, 2, 2, 2], kernel_size=[3, 3, 3, 3], num_classes=5, block='ConvNormAct',
                 pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2
        block = get_block(block)
        norm = get_norm(norm)

        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2 * base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0],
                                kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8 * base_ch, 10 * base_ch, num_block=num_block, block=block, pool=pool,
                                down_scale=scale[3], kernel_size=kernel_size[3], norm=norm)

        self.filterTrans3 = FilterTrans(4 * base_ch, 2 * base_ch,x1_img_size=[24,24,24],x2_img_size = [48,48,48])
        #self.filterTrans4 = FilterTrans(8 * base_ch, 4 * base_ch,x1_img_size=[12,12,12],x2_img_size = [24,24,24])

        self.up1 = up_block(10 * base_ch, 8 * base_ch, num_block=num_block, block=block, up_scale=scale[3],
                            kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8 * base_ch, 4 * base_ch, num_block=num_block, block=block, up_scale=scale[2],
                            kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4 * base_ch, 2 * base_ch, num_block=num_block, block=block, up_scale=scale[1],
                            kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2 * base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0],
                            kernel_size=kernel_size[0], norm=norm)

        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        t3 = self.filterTrans3(x3,x2)
        #t4 = self.filterTrans4(x4,x3)
        out = self.up1(x5, x4)
        out = self.up2(out, t3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)

        return out


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(1, 3, 96, 96, 96)).cuda()
    model = FilterTransU_3D(3, 32).cuda()
    param1 = sum([param.nelement() for param in model.parameters()])
    # param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    # print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    print(param1)

    # 统计每一层的参数数量和总参数数量
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(name, param.shape, num_params)
            total_params += num_params

    print("Total Parameters:", total_params)

    # 打印模型的层数和计算复杂度估计
    summary(model, (3, 96, 96, 96))

