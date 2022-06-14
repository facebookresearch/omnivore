#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py # NOQA
# and https://github.com/facebookresearch/deit/blob/main/models.py by Matthew
# Leavitt (ito@fb.com, matthew.l.leavitt@gmail.com) and Vedanuj Goswami
# (vedanuj@fb.com).

# FIXME: DO NOT OPEN SOURCE THIS - we're missing proper attributions!


import copy
import math
from functools import partial
from typing import List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.modules.utils import _ntuple


to_2tuple = _ntuple(2)


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PadIm2Video(torch.nn.Module):
    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__()
        self.time_dim = time_dim
        assert ntimes > 0
        assert pad_type in ["zero", "repeat"]
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x):
        if x.ndim == 4:
            # B, C, H, W -> B, C, T, H, W
            return x.unsqueeze(self.time_dim)
        elif x.ndim == 5:
            return x
        else:
            raise ValueError(f"Dimension incorrect {x.shape}")

        if x.shape[self.time_dim] == 1:
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
            elif self.pad_type == "zero":
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim]
                x = torch.nn.functional.pad(x, padarg)
        return x


class Mlp(nn.Module):
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


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = nn.functional.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = nn.functional.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        attn_target,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale_type = layer_scale_type

        # LVViT non-skip. Can be thought of as a special case of layer_scale
        if non_skip_wt_learnable is False:
            self.non_skip_wt = non_skip_wt
        else:
            self.non_skip_wt = nn.Parameter((torch.ones(1) * non_skip_wt).squeeze())

        # Layerscale
        if self.layer_scale_type is not None:
            # cannot use non_skip with this mode
            assert non_skip_wt == 1.0 and non_skip_wt_learnable is False
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.non_skip_wt)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.non_skip_wt)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale_gamma1)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.layer_scale_gamma2)
        return x

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_layout = (
            1,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = np.prod(self.patches_layout)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedConv(nn.Module):
    def __init__(self, conv_param_list, img_size=224, patch_size=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_layout = (
            1,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = np.prod(self.patches_layout)

        layers = []
        for idx, k in enumerate(conv_param_list):
            conv = nn.Conv2d(
                k["input_channels"],
                k["output_channels"],
                kernel_size=k["kernel_size"],
                stride=k["stride"],
                padding=k["padding"],
                bias=k["bias"],
            )
            layers.append(conv)
            if idx != len(conv_param_list) - 1:
                if k["norm"] == "bn":
                    norm = nn.BatchNorm2d(k["output_channels"])
                    layers.append(norm)
                elif k["norm"] == "lnfp32":
                    norm = Fp32GroupNorm(1, k["output_channels"])
                    layers.append(norm)
                elif k["norm"] == "ln":
                    norm = nn.GroupNorm(1, k["output_channels"])
                    layers.append(norm)
                if k["act"] == "relu":
                    act = nn.ReLU(inplace=True)
                    layers.append(act)
                elif k["act"] == "gelu":
                    act = nn.GELU()
                    layers.append(act)
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedGeneric(nn.Module):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, proj_stem, img_size):
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]
        # get the num_patches
        assert (
            isinstance(img_size, list) and len(img_size) >= 3
        ), "Need the full C[xT]xHxW in generic"
        # compute num_tokens with a forward
        with torch.no_grad():
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            )
            self.patches_layout = tuple(self.proj(dummy_img).shape[2:])
            self.num_patches = np.prod(self.patches_layout)

    def forward(self, x):
        # rgirdhar: no flatten here since the projection can handle it in the list of ops
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


class Decoder(nn.Module):
    def __init__(
        self,
        first_patch_idx,
        patches_layout,
        attn_target,
        embed_dim,
        decoder_embed_dim=512,
        decoder_depth=8,
        drop_path_rate=0.0,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_norm_eps=1e-6,
        non_skip_wt=1.0,
        return_interim_layers=False,
        share_pos_embed=False,
        learnable_pos_embed=True,
        init_pos_embed_random=False,
        non_skip_wt_learnable=False,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
        final_projection=None,
        pos_sum_embed_only=False,
        **kwargs,
    ):
        super().__init__()
        self.patches_layout = patches_layout
        self.first_patch_idx = first_patch_idx
        assert first_patch_idx == 0 or first_patch_idx == 1
        self.share_pos_embed = share_pos_embed
        self.build_pos_embedding(
            share_pos_embed=share_pos_embed,
            learnable_pos_embed=learnable_pos_embed,
            patches_layout=patches_layout,
            first_patch_idx=first_patch_idx,
            embed_dim=embed_dim,
            init_pos_embed_random=init_pos_embed_random,
        )
        self.pos_sum_embed_only = pos_sum_embed_only
        if pos_sum_embed_only:
            # another flag to catch if someone set this accidentally
            # recommended to use the `PosEmbedSumDecoder` class
            assert decoder_depth == -1, "Do not specify decoder_depth"
            return
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.norm = norm_layer(decoder_embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)
        ]  # stochastic depth decay rule

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    non_skip_wt=non_skip_wt,
                    non_skip_wt_learnable=non_skip_wt_learnable,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(decoder_depth)
            ]
        )
        self.return_interim_layers = return_interim_layers
        self.final_projection = None
        if final_projection is not None:
            self.final_projection = hydra.utils.instantiate(
                final_projection, _convert_="all", _recursive_=False
            )

    def build_pos_embedding(
        self,
        share_pos_embed,
        learnable_pos_embed,
        patches_layout,
        first_patch_idx,
        embed_dim,
        init_pos_embed_random,
    ):
        if share_pos_embed is True:
            # we expect pos_embed to be passed during `forward`
            # sharing nn.Parameter objects across modules is not recommended practice in PyTorch
            self.pos_embed = None
        elif learnable_pos_embed is True:
            self.pos_embed = nn.Parameter(
                # adding first_patch_idx since it is 1 if there is cls token, else 0
                torch.zeros(1, np.prod(patches_layout) + first_patch_idx, embed_dim)
            )
            if init_pos_embed_random:
                trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer(
                "pos_embed",
                get_sinusoid_encoding_table(
                    np.prod(patches_layout) + first_patch_idx, embed_dim
                ),
            )

    def forward(
        self,
        x,
        orig_input_shape,
        input_pos_embed=None,
        use_checkpoint=False,
    ):
        curr_pos_embed = input_pos_embed if self.share_pos_embed else self.pos_embed
        pos_embed = VisionTransformer.get_pos_embedding(
            x.size(1) - self.first_patch_idx,
            curr_pos_embed,
            self.patches_layout,
            input_shape=orig_input_shape,
            first_patch_idx=self.first_patch_idx,
        )
        x = x + pos_embed
        if self.pos_sum_embed_only:
            return x
        x = self.decoder_embed(x)
        interim = []
        for i, blk in enumerate(self.decoder_blocks):
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if self.return_interim_layers or i == (len(self.decoder_blocks) - 1):
                interim.append(x)

        interim = [self.norm(el) for el in interim]
        if self.final_projection is not None:
            interim = [self.final_projection(el) for el in interim]
        if self.return_interim_layers:
            return interim
        return interim[-1]


class PosEmbedSumDecoder(Decoder):
    def __init__(
        self,
        first_patch_idx,
        patches_layout,
        embed_dim,
        share_pos_embed=False,
        learnable_pos_embed=True,
        init_pos_embed_random=False,
    ):
        super().__init__(
            pos_sum_embed_only=True,
            decoder_depth=-1,
            attn_target=None,
            first_patch_idx=first_patch_idx,
            patches_layout=patches_layout,
            embed_dim=embed_dim,
            share_pos_embed=share_pos_embed,
            learnable_pos_embed=learnable_pos_embed,
            init_pos_embed_random=init_pos_embed_random,
        )


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        attn_target,
        embed_dim,
        num_blocks,
        drop_path_rate=0.0,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        non_skip_wt=1.0,
        attn_drop_rate=0.0,
        layer_norm_eps=1e-6,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=1e-4,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.norm = norm_layer(embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    non_skip_wt=non_skip_wt,
                    non_skip_wt_learnable=non_skip_wt_learnable,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x, use_checkpoint=False):
        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision transformer. Adding stochastic depth makes it a DeiT.
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        mlp_ratio,
        attn_target,
        drop_rate,
        drop_path_rate,
        drop_path_type,
        force_cast_ln_fp32,
        classifier_feature,
        use_cls_token,
        learnable_pos_embed,
        non_skip_wt,
        non_skip_wt_learnable,
        layer_scale_type,
        layer_scale_init_value,
        patch_embed_type,
        patch_embed_params_list,
        layer_norm_eps=1e-6,
        masked_image_modeling=False,
        patch_drop_min_patches=-1,
        patch_drop_max_patches=-1,
        patch_drop_at_eval=False,
        add_pos_same_dtype=False,
        patch_dropping=False,
        post_encoder_params=None,
        decoder=None,
        mask_token_embed_dim=None,
    ):
        super().__init__()

        assert use_cls_token or classifier_feature == "global_pool"
        self.masked_image_modeling = masked_image_modeling
        self.patch_drop_min_patches = patch_drop_min_patches
        self.patch_drop_max_patches = patch_drop_max_patches
        self.patch_drop_at_eval = patch_drop_at_eval

        self.add_pos_same_dtype = add_pos_same_dtype

        # turn off mae masking for eval
        self.patch_dropping = patch_dropping

        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        if force_cast_ln_fp32:
            norm_layer = partial(Fp32LayerNorm, eps=layer_norm_eps)

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        assert classifier_feature in ["cls_token", "global_pool"]
        self.classifier_feature = classifier_feature

        assert in_chans == 3, "Only 3 channels supported"

        if patch_embed_type == "linear":
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        elif patch_embed_type == "conv":
            self.patch_embed = PatchEmbedConv(
                conv_param_list=patch_embed_params_list,
                img_size=img_size,
                patch_size=patch_size,
            )

        elif patch_embed_type == "generic":

            self.patch_embed = PatchEmbedGeneric(
                patch_embed_params_list, img_size=img_size
            )

        num_patches = self.patch_embed.num_patches
        assert (
            self.patch_embed.patches_layout[-1] == self.patch_embed.patches_layout[-2]
        ), "Interpolation of pos embed not supported for non-square layouts"

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.first_patch_idx = 1
            total_num_patches = num_patches + 1
        else:
            self.cls_token = None
            self.first_patch_idx = 0
            total_num_patches = num_patches

        if learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, total_num_patches, embed_dim))
        else:
            self.register_buffer(
                "pos_embed", get_sinusoid_encoding_table(total_num_patches, embed_dim)
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        assert drop_path_type in [
            "progressive",
            "uniform",
        ], f"Drop path types are: [progressive, uniform]. Got {drop_path_type}."
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    non_skip_wt=non_skip_wt,
                    non_skip_wt_learnable=non_skip_wt_learnable,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(depth)
            ]
        )

        # FIXME: Verify if we use Post encoder, if not, remove it.
        self.post_encoder = None

        if post_encoder_params is not None:
            self.post_encoder = hydra.utils.instantiate(
                post_encoder_params,
                _convert_="all",
            )

        if self.patch_dropping and decoder is None:
            self.decoder = None

        if mask_token_embed_dim is None:
            mask_token_embed_dim = embed_dim

        if decoder is not None:
            self.decoder = decoder(
                first_patch_idx=self.first_patch_idx,
                patches_layout=self.patch_embed.patches_layout,
                embed_dim=mask_token_embed_dim,
            )

        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        if learnable_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
        if self.patch_dropping and patch_embed_type == "linear":
            # Based on MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

        if self.masked_image_modeling:
            assert self.patch_drop_max_patches == -1
            # initialized to zeros following iBOT
            self.mask_token = nn.Parameter(torch.zeros(1, mask_token_embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.patch_dropping:  # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, Fp32LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def patch_drop(self, x, npatch_per_img, patch_start_idx=1, npatch_to_keep=None):
        """
        Randomly drop patches from the input
        Input:
            - x: B x N x C
        Returns:
            - y: B x N' x C where N' is sampled from [self.patch_drop_min_patches, self.patch_drop_max_patches]
        """
        if (
            self.patch_drop_min_patches < 0
            or self.patch_drop_min_patches == npatch_per_img
            or (npatch_to_keep is not None and npatch_to_keep < 0)
        ):
            return x

        # typically we do not drop patches at test time.
        # controlled by a flag `patch_drop_at_eval`
        # we may want to drop patches in eval mode for self-supervised teachers
        if self.training is False and self.patch_drop_at_eval is False:
            return x

        rnd_inds = [
            torch.randperm(npatch_per_img, device=x.device) for _ in range(x.shape[0])
        ]

        if npatch_to_keep is None:
            npatch_to_keep = torch.randint(
                low=self.patch_drop_min_patches,
                high=self.patch_drop_max_patches,
                size=(1,),
            ).item()
        class_tokens = x[:, :patch_start_idx, ...]
        patch_tokens = x[:, patch_start_idx:, ...]

        patch_tokens = [
            patch_tokens[i, rnd_inds[i][:npatch_to_keep]] for i in range(x.shape[0])
        ]
        patch_tokens = torch.stack(patch_tokens)
        x = torch.cat([class_tokens, patch_tokens], dim=1)
        return x

    def masked_patch_drop(self, x, mask):
        mask = mask.view(x.shape[0], -1)
        cls_token, patches = (
            x[:, : self.first_patch_idx],
            x[:, self.first_patch_idx :],
        )
        # If the following fails, means different batch elements have
        # different amounts of masking. To fix ensure the masking amount
        # is fixed in the config for all datasets being trained on.
        patches = patches[~mask].reshape(x.shape[0], -1, patches.shape[-1])
        x = torch.cat([cls_token, patches], dim=1)
        return x

    def apply_mask(self, x, mask):
        mask = mask.view(x.shape[0], -1)
        x[mask, :] = self.mask_token.to(x.dtype)
        return x

    def insert_masks(self, x, mask):
        embed_dim = x.shape[-1]
        mask = mask.view(x.shape[0], -1)
        B, N = mask.shape
        tmp = torch.empty(B, N, embed_dim).to(x.device)
        tmp[mask] = self.mask_token.to(x.dtype)
        tmp[~mask] = x[:, self.first_patch_idx :].reshape(-1, x.shape[-1])
        x = torch.cat([x[:, : self.first_patch_idx], tmp], dim=1)
        return x

    def prepare_tokens(self, x, npatch_to_keep, mask):
        B = x.shape[0]
        input_shape = x.shape

        x = self.patch_embed(x)
        npatch_per_img = x.shape[1]

        if self.patch_dropping is False and mask is not None:
            x = self.apply_mask(x, mask)

        if self.cls_token is not None:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            x = torch.cat((class_tokens, x), dim=1)

        pos_embed = self.get_pos_embedding(
            npatch_per_img,
            self.pos_embed,
            self.patch_embed.patches_layout,
            input_shape,
            first_patch_idx=self.first_patch_idx,
        )
        if self.add_pos_same_dtype:
            pos_embed = pos_embed.type_as(x)
        x = x + pos_embed

        if self.patch_dropping and mask is not None:
            x = self.masked_patch_drop(x, mask)
        else:
            x = self.patch_drop(
                x,
                npatch_per_img,
                patch_start_idx=self.first_patch_idx,
                npatch_to_keep=npatch_to_keep,
            )
        x = self.pos_drop(x)
        return x

    @classmethod
    def get_pos_embedding(
        cls,
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape,
        first_patch_idx=1,
    ):
        pos_embed = cls.interpolate_pos_encoding(
            npatch_per_img,
            pos_embed,
            patches_layout,
            input_shape=input_shape,
            first_patch_idx=first_patch_idx,
        )
        return pos_embed

    def forward_features(self, x, npatch_to_keep, mask=None, use_checkpoint=False):
        assert npatch_to_keep is None
        if mask is not None and isinstance(mask, list) and not all(mask):
            mask = None

        orig_input_shape = x.shape
        x = self.prepare_tokens(x, npatch_to_keep, mask=mask)

        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.classifier_feature == "cls_token" and (
            mask is None or self.decoder is None
        ):
            assert self.first_patch_idx == 1, "Must have a CLS token at 0"
            x = x[:, 0]
        elif self.classifier_feature == "global_pool" and (
            mask is None or self.decoder is None
        ):
            x = x[:, self.first_patch_idx :, ...].mean(dim=1)
        elif self.patch_dropping and mask is not None and self.decoder is not None:
            x = self.norm(x)
            if self.post_encoder:
                x_dtype = x.dtype
                x = self.post_encoder(x).to(x_dtype)
            x = self.insert_masks(x, mask)
            if self.first_patch_idx == 0:
                cls_token = None
            else:
                cls_token = x[:, self.first_patch_idx]
            x = self.decoder(
                x, orig_input_shape, self.pos_embed, use_checkpoint=use_checkpoint
            )
            # cls_token comes from the encoder and the x comes from
            # the decoder. Since they can have different dimensionality, they
            # can't be concatenated together and have to be returned as a tuple

            if isinstance(x, list):
                decoder_patch_features = [el[:, self.first_patch_idx :] for el in x]
            else:
                decoder_patch_features = x[:, self.first_patch_idx :]
            return cls_token, decoder_patch_features
        elif mask is not None:
            pass

        x = self.norm(x)
        return self.pre_logits(x)

    def get_intermediate_features(
        self, x, names, npatch_to_keep, mask, use_checkpoint=False
    ):
        interms = []

        x = self.prepare_tokens(x, npatch_to_keep, mask=mask)

        # get feature from every intermediate block and apply norm
        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            interms.append(self.norm(x))

        if self.post_encoder:
            assert len(names) == 1 and names[0] in ["last_all"]
            interms.append(self.post_encoder(interms[-1]))

        # feature names are as follows
        # blkCLS[integer] => CLS token of blk[integer]
        # concatCLS[integer] => concat of CLS token from last "integer" blocks
        # lastCLS => CLS token of last block

        output = []

        for name in names:
            if name.startswith("blkCLS"):
                assert self.first_patch_idx == 1, "Must have CLS token at 0"
                v = int(name.replace("blkCLS", ""))
                output.append(interms[v][:, 0])
            elif name.startswith("concatCLS"):
                assert self.first_patch_idx == 1, "Must have CLS token at 0"
                v = int(name.replace("concatCLS", ""))
                feat = torch.cat([x[:, 0] for x in interms[-v:]], dim=-1)
                output.append(feat)
            elif name == "lastCLS":
                assert self.first_patch_idx == 1, "Must have CLS token at 0"
                output.append(interms[-1][:, 0])
            elif name == "last_all":
                output.append(interms[-1])
            elif name == "last_patch_avg":
                output.append(interms[-1][:, self.first_patch_idx :, ...].mean(dim=1))
        return output

    def forward(
        self,
        x: torch.Tensor,
        out_feat_keys: List[str] = None,
        npatch_to_keep: int = None,
        mask: torch.Tensor = None,
        use_checkpoint: bool = False,
    ) -> List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            x = self.forward_features(
                x, npatch_to_keep, mask=mask, use_checkpoint=use_checkpoint
            )
            if not isinstance(x, tuple):
                x = x.unsqueeze(0)
            else:
                # This happens when forward_features returns a tuple with the
                # cls_token and features in a tuple instead of concatenated together.
                # This happens when there is a decoder which might decode the features
                # into a different dimensionality than the encoder
                x = [x]
        else:
            # we specified a feature layer name
            # Follow DINO (https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L159)
            x = self.get_intermediate_features(
                x,
                out_feat_keys,
                npatch_to_keep,
                mask=mask,
                use_checkpoint=use_checkpoint,
            )
        return x

    @staticmethod
    def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
        N = pos_embed.shape[1]
        if N == target_spatial_size:
            return pos_embed
        dim = pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(target_spatial_size / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    @classmethod
    def interpolate_pos_encoding(
        cls,
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=None,
        first_patch_idx=1,
    ):
        assert (
            first_patch_idx == 0 or first_patch_idx == 1
        ), "there is 1 CLS token or none"
        N = pos_embed.shape[1] - first_patch_idx  # since it's 1 if cls_token exists
        if npatch_per_img == N:
            return pos_embed
        class_emb = pos_embed[:, :first_patch_idx]
        pos_embed = pos_embed[:, first_patch_idx:]

        if input_shape is None or patches_layout[0] == 1:
            # simple 2D pos embedding, no temporal component
            pos_embed = cls.interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
        elif patches_layout[0] > 1:
            # pos embed has a temporal component
            assert len(input_shape) == 4, "temporal interpolation not supported"
            # we only support 2D interpolation in this case
            num_frames = patches_layout[0]
            num_spatial_tokens = patches_layout[1] * patches_layout[2]
            pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
            # interpolate embedding for zeroth frame
            pos_embed = cls.interpolate_pos_encoding_2d(
                npatch_per_img, pos_embed[0, 0, ...].unsqueeze(0)
            )
        else:
            raise ValueError("This type of interpolation isn't implemented")

        return torch.cat((class_emb, pos_embed), dim=1)

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()
        if layer_name in ["cls_token", "pos_embed"]:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers

    def get_num_layers(self):
        return len(self.blocks) + 1
