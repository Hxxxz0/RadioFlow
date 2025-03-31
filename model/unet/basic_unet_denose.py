# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps = timesteps * 1000  # Adjust the scaling factor according to specific needs
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 平均池化后通过FC
        max_out = self.fc(self.max_pool(x))  # 最大池化后通过FC
        out = avg_out + max_out  # 融合两种池化结果
        return self.sigmoid(out)  # 输出注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大值
        out = torch.cat([avg_out, max_out], dim=1)  # 拼接
        out = self.conv(out)  # 卷积生成空间注意力图
        return self.sigmoid(out)  # 输出注意力权重

# 改进的CrossAttention模块
class CrossAttention(nn.Module):
    def __init__(self, in_channels, embedding_channels, reduction=16, kernel_size=7):
        super(CrossAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.embedding_proj = nn.Conv2d(embedding_channels, in_channels, 1)  # 投影嵌入特征

    def forward(self, x, embedding):
        # 将嵌入特征投影到与特征图相同的通道数
        embedding_proj = self.embedding_proj(embedding)
        
        # 应用通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca  # 加权特征图
        
        # 应用空间注意力
        sa = self.spatial_attention(x)
        x_sa = x * sa  # 加权特征图
        
        # 结合通道和空间注意力
        x_attn = x_ca + x_sa
        
        # 与投影后的嵌入特征融合
        fused = x_attn + embedding_proj
        
        return fused

class CrossAttention_old(nn.Module):
    """Cross-attention module for interaction between feature maps and embeddings"""
    def __init__(self, dim, num_heads=8, head_dim=None, dropout=0.0, downsample_factor=4):
        """
        Args:
            dim: Number of channels in the input features
            num_heads: Number of heads in multi-head attention
            head_dim: Dimension of each head, calculated automatically if None
            dropout: Dropout ratio
            downsample_factor: Spatial dimension downsampling factor to reduce memory consumption
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.downsample_factor = downsample_factor
        
        # Features as query
        self.q_proj = nn.Linear(dim, num_heads * head_dim)
        # Embedding as key and value
        self.k_proj = nn.Linear(dim, num_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_heads * head_dim)
        
        self.out_proj = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, embedding):
        """
        Args:
            x: Feature map [B, C, ...] (can be 2D or 3D data)
            embedding: Embedding feature map [B, C, ...] (with the same spatial dimensions as x)
        Returns:
            Fused feature map with the same shape as input x
        """
        # Get batch size and channel count
        batch_size = x.shape[0]
        channels = x.shape[1]
        original_shape = x.shape
        
        # Check input dimensions, adjust embedding shape
        if embedding.dim() != x.dim():
            # If embedding's dimension differs from x, adjustments are needed
            if embedding.dim() > x.dim():
                # Embedding has higher dimension, flatten it to match x's dimension
                embedding = embedding.reshape(batch_size, embedding.shape[1], -1).mean(dim=2)
                # Then expand to match x's spatial dimensions
                new_shape = list(x.shape)
                new_shape[1] = embedding.shape[1]
                embedding = embedding.unsqueeze(-1).unsqueeze(-1).expand(new_shape)
            else:
                # Embedding has lower dimension, expand to match x's dimensions
                new_shape = list(x.shape)
                new_shape[1] = embedding.shape[1]  # Keep channel count unchanged
                embedding = embedding.view(*embedding.shape, *([1] * (x.dim() - embedding.dim()))).expand(new_shape)
        
        # Downsample high-resolution features to reduce computation
        if self.downsample_factor > 1 and x.dim() > 3:  # Only downsample features with spatial dimensions > 1
            # Calculate average spatial dimension size
            spatial_size = 1
            for i in range(2, x.dim()):
                spatial_size *= x.shape[i]
            spatial_size = int(spatial_size ** (1/(x.dim()-2)))  # Geometric mean
            
            # If spatial size exceeds a threshold, perform downsampling
            if spatial_size > 32:  # Threshold can be adjusted based on actual needs
                # Use adaptive average pooling for downsampling
                target_size = [max(1, s // self.downsample_factor) for s in x.shape[2:]]
                if x.dim() == 4:  # 2D data
                    pool = nn.AdaptiveAvgPool2d(target_size)
                elif x.dim() == 5:  # 3D data
                    pool = nn.AdaptiveAvgPool3d(target_size)
                else:
                    # Generic processing approach
                    x_down = x.reshape(batch_size, channels, -1)
                    x_down = x_down[:, :, ::self.downsample_factor**2]
                    embedding_down = embedding.reshape(batch_size, embedding.shape[1], -1)
                    embedding_down = embedding_down[:, :, ::self.downsample_factor**2]
                    
                    # Flatten feature maps into sequences
                    x_flat = x_down.permute(0, 2, 1)  # [B, L_down, C]
                    emb_flat = embedding_down.permute(0, 2, 1)  # [B, L_down, C]
                    
                    # Subsequent calculations are the same as the original
                    seq_len = x_flat.shape[1]
                    
                    # Calculate query, key, value
                    q = self.q_proj(x_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
                    k = self.k_proj(emb_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
                    v = self.v_proj(emb_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
                    
                    # Calculate attention scores
                    attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, L, L]
                    attn = attn.softmax(dim=-1)
                    attn = self.dropout(attn)
                    
                    # Apply attention to get output
                    out = (attn @ v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.head_dim)  # [B, L, C]
                    out = self.out_proj(out)
                    
                    # Reshape to downsampled shape, then upsample
                    down_shape = list(target_size)
                    down_shape.insert(0, channels)
                    down_shape.insert(0, batch_size)
                    out = out.permute(0, 2, 1).reshape(down_shape) if len(target_size) > 1 else out.permute(0, 2, 1)
                    
                    # Upsample back to original size
                    if x.dim() == 4:  # 2D data
                        out = nn.functional.interpolate(out, size=original_shape[2:], mode='bilinear', align_corners=False)
                    elif x.dim() == 5:  # 3D data
                        out = nn.functional.interpolate(out, size=original_shape[2:], mode='trilinear', align_corners=False)
                    else:
                        # Generic processing
                        out_flat = out.permute(0, 2, 1)
                        out = nn.functional.interpolate(out_flat.unsqueeze(1), size=original_shape[2], mode='linear').squeeze(1)
                        out = out.permute(0, 2, 1).reshape(original_shape)
                    
                    return out
                
                # Downsample
                x_down = pool(x)
                embedding_down = pool(embedding)
                
                # Flatten feature maps into sequences
                x_flat = x_down.reshape(batch_size, channels, -1).permute(0, 2, 1)  # [B, L_down, C]
                emb_flat = embedding_down.reshape(batch_size, embedding.shape[1], -1).permute(0, 2, 1)  # [B, L_down, C]
                
                seq_len = x_flat.shape[1]
                
                # Calculate query, key, value
                q = self.q_proj(x_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
                k = self.k_proj(emb_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
                v = self.v_proj(emb_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
                
                # Calculate attention scores
                attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, L, L]
                attn = attn.softmax(dim=-1)
                attn = self.dropout(attn)
                
                # Apply attention to get output
                out = (attn @ v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.head_dim)  # [B, L, C]
                out = self.out_proj(out)
                
                # Reshape to downsampled shape, then upsample
                down_shape = list(x_down.shape)
                out = out.permute(0, 2, 1).reshape(down_shape)
                
                # Upsample back to original size
                if x.dim() == 4:  # 2D data
                    out = nn.functional.interpolate(out, size=original_shape[2:], mode='bilinear', align_corners=False)
                elif x.dim() == 5:  # 3D data
                    out = nn.functional.interpolate(out, size=original_shape[2:], mode='trilinear', align_corners=False)
                
                return out
        
        # Standard processing flow (no downsampling)
        # Flatten feature maps into sequences
        x_flat = x.reshape(batch_size, channels, -1).permute(0, 2, 1)  # [B, L, C]
        seq_len = x_flat.shape[1]
        
        # Flatten embedding into sequences as well
        emb_c = embedding.shape[1]
        emb_flat = embedding.reshape(batch_size, emb_c, -1).permute(0, 2, 1)  # [B, L, C]
        
        # If sequence lengths differ, adjust embedding's sequence length
        if emb_flat.shape[1] != seq_len:
            # Use interpolation or other methods to adjust length, here simply use repeat/truncate
            if emb_flat.shape[1] < seq_len:
                # Repeat to target length
                emb_flat = emb_flat.repeat_interleave(seq_len // emb_flat.shape[1] + 1, dim=1)[:, :seq_len, :]
            else:
                # Truncate to target length
                emb_flat = emb_flat[:, :seq_len, :]
        
        # Calculate query, key, value
        q = self.q_proj(x_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
        k = self.k_proj(emb_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
        v = self.v_proj(emb_flat).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, L, hd]
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, L, L]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to get output
        out = (attn @ v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.head_dim)  # [B, L, C]
        out = self.out_proj(out)
        
        # Reshape back to original shape
        out = out.permute(0, 2, 1).reshape(original_shape)  # [B, C, ...]
        
        return out


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        x = self.conv_0(x) 
        # Change temb shape from [B, C] to [B, C, 1, 1]
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv_1(x)
        return x

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x 

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, temb)

        return x


class BasicUNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        use_cross_attention: bool = True,
        attention_heads: int = 4,
        attention_downsample: int = 8,
        dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            use_cross_attention: 是否使用交叉注意力机制代替简单的嵌入加法。默认为True。
            attention_heads: 交叉注意力中的注意力头数量。默认为4。
            attention_downsample: 交叉注意力中的下采样因子，用于减少内存消耗。默认为8。

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        self.use_cross_attention = use_cross_attention
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        
        # If using cross-attention, create cross-attention modules for each layer
        if self.use_cross_attention:
            self.cross_attn_0 = CrossAttention(fea[0], fea[0], reduction=16, kernel_size=7)
            self.cross_attn_1 = CrossAttention(fea[1], fea[1], reduction=16, kernel_size=7)
            self.cross_attn_2 = CrossAttention(fea[2], fea[2], reduction=16, kernel_size=7)
            self.cross_attn_3 = CrossAttention(fea[3], fea[3], reduction=16, kernel_size=7)
            self.cross_attn_4 = CrossAttention(fea[4], fea[4], reduction=16, kernel_size=7)

    def forward(self, x: torch.Tensor, t, embeddings=None, image=None):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.
            t: 时间步
            embeddings: 条件嵌入特征，如果为None则不使用条件
            image: 额外输入的图像，将与x拼接作为输入

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)
            
        x0 = self.conv_0(x, temb)
        if embeddings is not None:
            if self.use_cross_attention:
                x0 = self.cross_attn_0(x0, embeddings[0])
            else:
                x0 += embeddings[0]

        x1 = self.down_1(x0, temb)
        if embeddings is not None:
            if self.use_cross_attention:
                x1 = self.cross_attn_1(x1, embeddings[1])
            else:
                x1 += embeddings[1]

        x2 = self.down_2(x1, temb)
        if embeddings is not None:
            if self.use_cross_attention:
                x2 = self.cross_attn_2(x2, embeddings[2])
            else:
                x2 += embeddings[2]

        x3 = self.down_3(x2, temb)
        if embeddings is not None:
            if self.use_cross_attention:
                x3 = self.cross_attn_3(x3, embeddings[3])
            else:
                x3 += embeddings[3]

        x4 = self.down_4(x3, temb)
        if embeddings is not None:
            if self.use_cross_attention:
                x4 = self.cross_attn_4(x4, embeddings[4])
            else:
                x4 += embeddings[4]

        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits



