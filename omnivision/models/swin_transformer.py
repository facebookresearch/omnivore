# Portions copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copied and modified from
# https://raw.githubusercontent.com/SwinTransformer/Video-Swin-Transformer/master/mmaction/models/backbones/swin_transformer.py

import logging
from functools import lru_cache, reduce
from operator import mul
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from omnivision.utils.checkpoint import load_and_broadcast_checkpoint_list
from timm.models.layers import DropPath, trunc_normal_


class Im2Video(nn.Module):
    """Convert an image into a trivial video."""

    def __init__(self, time_dim=2):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, x):
        if x.ndim == 4:
            # B, C, H, W -> B, C, T, H, W
            return x.unsqueeze(self.time_dim)
        elif x.ndim == 5:
            return x
        else:
            raise ValueError(f"Dimension incorrect {x.shape}")


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )
    return windows


def window_partition_image(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[1], window_size[1], W // window_size[2], window_size[2], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[1], window_size[2], C)
    )
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w)
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(
            N, N, -1
        )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        )  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix, use_checkpoint=False):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if use_checkpoint:
            x = checkpoint.checkpoint(
                self.forward_part1, x, mask_matrix, use_reentrant=False
            )
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H=None, W=None):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        if H is None:
            B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, use_checkpoint=False, H=None, W=None, use_seg=False):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        if use_seg:
            return self.forward_seg(x, H, W)
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask, use_checkpoint=use_checkpoint)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

    def forward_seg(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]),
            slice(-self.shift_size[2], None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_image(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if x.ndim == 4:
                B, D, C, L = x.shape
                assert L == H * W, "input feature has wrong size"
                x = x.reshape(B, D, C, H, W)
                x = x.permute(0, 1, 3, 4, 2)
            assert x.shape[2] == H
            assert x.shape[3] == W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        additional_variable_channels=None,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.additional_variable_channels = additional_variable_channels

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if additional_variable_channels:
            # we create var_proj separately from proj
            # this makes it convenient to ignore var_proj on downstream tasks
            # where we only use RGB
            self.var_proj = [
                nn.Conv3d(x, embed_dim, kernel_size=patch_size, stride=patch_size)
                for x in additional_variable_channels
            ]
            self.var_proj = nn.ModuleList(self.var_proj)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def run_variable_channel_forward(self, x):
        sidx = 0
        out = None
        for idx in range(len(self.additional_variable_channels)):
            eidx = sidx + self.additional_variable_channels[idx]
            c_out = self.var_proj[idx](x[:, sidx:eidx, ...])
            if idx == 0:
                out = c_out
            else:
                out += c_out
            sidx = eidx
        return out

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        if self.additional_variable_channels:
            x_rgb = x[:, :3, ...]
            x_rem = x[:, 3:, ...]
            x_rgb = self.proj(x_rgb)
            if x.shape[1] > 3:
                x_rem = self.run_variable_channel_forward(x_rem)
                x = x_rgb + x_rem
            else:
                x = x_rgb
        else:
            x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class SwinTransformer3D(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
        self,
        pretrained=None,
        pretrained2d=True,
        pretrained3d=None,
        pretrained_model_key="base_model",
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        frozen_stages=-1,
        depth_mode=None,
        depth_patch_embed_separate_params=True,
    ):
        super().__init__()

        self.im2vid = Im2Video()
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.pretrained3d = pretrained3d
        self.pretrained_model_key = pretrained_model_key
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        self.depth_mode = depth_mode
        depth_chans = None
        assert in_chans == 3, "Only 3 channels supported"

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if depth_mode is not None:
            msg = f"Using depth mode {depth_mode}"
            logging.info(msg)
            assert depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens", "rgbd"]
            if depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens"]:
                depth_chans = 1
                assert (
                    depth_patch_embed_separate_params
                ), "separate tokenization needs separate parameters"
                if depth_mode == "separate_d_tokens":
                    raise NotImplementedError()
            else:
                assert depth_mode == "rgbd"
                depth_chans = 4

            self.depth_patch_embed_separate_params = depth_patch_embed_separate_params

            if depth_patch_embed_separate_params:
                self.depth_patch_embed = PatchEmbed3D(
                    patch_size=patch_size,
                    in_chans=depth_chans,
                    embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None,
                )
            else:
                # share parameters with patch_embed
                # delete the layer we built above
                del self.patch_embed
                assert depth_chans == 4
                logging.info(
                    "Certain channels of patch projection may not be used in forward pass"
                )
                logging.info(
                    "Make sure config.DISTRIBUTED.FIND_UNUSED_PARAMETERS is set to True"
                )
                self.patch_embed = PatchEmbed3D(
                    patch_size=patch_size,
                    in_chans=3,
                    embed_dim=embed_dim,
                    additional_variable_channels=[1],
                    norm_layer=norm_layer if self.patch_norm else None,
                )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        # init the weights
        self.init_weights()

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = load_and_broadcast_checkpoint_list(self.pretrained)

        if "classy_state_dict" in checkpoint:
            # checkpoints trained in omnivore
            state_dict = checkpoint["classy_state_dict"][self.pretrained_model_key][
                "model"
            ]["trunk"]
        else:
            # checkpoints trained outside omnivore
            state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if "relative_position_index" in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        if state_dict["patch_embed.proj.weight"].ndim == 4:
            state_dict["patch_embed.proj.weight"] = state_dict[
                "patch_embed.proj.weight"
            ].unsqueeze(2)
        state_dict["patch_embed.proj.weight"] = (
            state_dict["patch_embed.proj.weight"].repeat(1, 1, self.patch_size[0], 1, 1)
            / self.patch_size[0]
        )
        if (
            "depth_patch_embed.proj.weight" in state_dict
            and state_dict["depth_patch_embed.proj.weight"].ndim == 4
        ):
            state_dict["depth_patch_embed.proj.weight"] = state_dict[
                "depth_patch_embed.proj.weight"
            ].unsqueeze(2)

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if "relative_position_bias_table" in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1**0.5)
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(
                                1, nH1, S1, S1
                            ),
                            size=(
                                2 * self.window_size[1] - 1,
                                2 * self.window_size[2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.view(
                            nH2, L2
                        ).permute(1, 0)
                    )
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1
            )
        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def load_and_interpolate_3d_weights(self, logger):
        checkpoint = load_and_broadcast_checkpoint_list(self.pretrained)
        assert self.pretrained3d is not None and self.pretrained2d is False

        if "classy_state_dict" in checkpoint:
            # checkpoints trained in omnivore
            state_dict = checkpoint["classy_state_dict"][self.pretrained_model_key][
                "model"
            ]["trunk"]
        else:
            # checkpoints trained outside omnivore
            state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if "relative_position_index" in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if "relative_position_bias_table" in k
        ]
        pretrained_window_size = self.pretrained3d
        T1 = 2 * pretrained_window_size[0] - 1
        S11 = 2 * pretrained_window_size[1] - 1
        S12 = 2 * pretrained_window_size[2] - 1
        assert (
            pretrained_window_size[0] == self.window_size[0]
        ), "Interpolating along time not supported"

        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (
                (2 * self.window_size[0] - 1)
                * (2 * self.window_size[1] - 1)
                * (2 * self.window_size[2] - 1)
            )
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    pretrained_bias = relative_position_bias_table_pretrained.view(
                        T1, S11, S12, nH1
                    )
                    pretrained_bias = pretrained_bias.permute(0, 3, 1, 2)
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            pretrained_bias,
                            size=(
                                2 * self.window_size[1] - 1,
                                2 * self.window_size[2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained_resized = (
                        relative_position_bias_table_pretrained_resized.permute(
                            0, 2, 3, 1
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.reshape(L2, nH2)
                    )

            state_dict[k] = relative_position_bias_table_pretrained
        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str) or isinstance(self.pretrained, list):
            self.apply(_init_weights)
            logging.info(f"load model from: {self.pretrained}")

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                logging.info(f"Inflating with {self.pretrained_model_key}")
                self.inflate_weights(logging)
            elif self.pretrained3d:
                logging.info(f"Loading 3D model with {self.pretrained_model_key}")
                self.load_and_interpolate_3d_weights(logging)
            else:
                raise ValueError(
                    "Use VISSL loading for this. This code "
                    "is only for Swin inflation."
                )
                # # Directly load 3D model.
                # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError(
                f"pretrained must be a str or None but found: {type(self.pretrained)}"
            )

    def _apply_norm(self, x):
        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")
        return x

    def forward_intermediate_features(self, stage_outputs, out_feat_keys):
        """
        Inputs
        - stage_outputs: list of features without self.norm() applied to them
        - out_feat_keys: list of feature names (str)
                         specified as "stage<int>" for feature with norm
                         or "interim<int>" for feature without norm
        """
        out_features = []
        for key in out_feat_keys:
            if key.startswith("stage"):
                rep = "stage"
            elif key.startswith("interim"):
                rep = "interim"
            else:
                raise ValueError(f"Invalid key {key}")
            idx = int(key.replace(rep, ""))
            feat = stage_outputs[idx]
            if rep == "stage":
                feat = self._apply_norm(feat)
            out_features.append(feat)
        return out_features

    def get_patch_embedding(self, x):
        # x: B x C x T x H x W
        assert x.ndim == 5
        has_depth = x.shape[1] == 4

        if has_depth:
            if self.depth_mode in ["summed_rgb_d_tokens"]:
                x_rgb = x[:, :3, ...]
                x_d = x[:, 3:, ...]
                x_d = self.depth_patch_embed(x_d)
                x_rgb = self.patch_embed(x_rgb)
                # sum the two sets of tokens
                x = x_rgb + x_d
            elif self.depth_mode == "rgbd":
                if self.depth_patch_embed_separate_params:
                    x = self.depth_patch_embed(x)
                else:
                    x = self.patch_embed(x)
            else:
                logging.info("Depth mode %s not supported" % self.depth_mode)
                raise NotImplementedError()
        else:
            x = self.patch_embed(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        out_feat_keys: List[str] = None,
        use_checkpoint: bool = False,
    ) -> List[torch.Tensor]:
        """Forward function."""
        # Convert to singleton video if not already
        x = self.im2vid(x)

        x = self.get_patch_embedding(x)

        x = self.pos_drop(x)

        stage_outputs = []

        for layer in self.layers:
            x = layer(x.contiguous(), use_checkpoint=use_checkpoint)
            stage_outputs.append(x)

        if out_feat_keys is not None and len(out_feat_keys) > 0:
            return self.forward_intermediate_features(stage_outputs, out_feat_keys)
        else:
            x = self._apply_norm(x)
            # Mean over the spatiotemporal dimensions
            x = torch.mean(x, [-3, -2, -1])

            return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()
