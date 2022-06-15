#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vision_transformer import VisionTransformer, Attention, Decoder, PadIm2Video


def make_conv_or_linear(layer, init_weight=None, init_bias=None):
    if init_weight is not None:
        init_weight(tensor=layer.weight.data)
    if init_bias is not None:
        init_bias(tensor=layer.bias.data)
    return layer


def reshape_and_init_as_mlp(tensor):
    # Based on MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
    torch.nn.init.xavier_uniform_(tensor.view([tensor.shape[0], -1]))


class OmniMAE(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, imgOrVideo, mask=None):
        # imgOrVideo: A tensor of shape [N,C,H,W] for images and [N,C,T,H,W] for videos
        # mask: A boolean tensor of the shape [N, patch_layout's shpae]
        outputs = self.trunk(imgOrVideo, mask=mask)
        if mask is not None:
            return self.head(outputs[0][1])
        else:
            return self.head(outputs[0])


def vit_base_mae_pretraining(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=12,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=768,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
        layer_norm_eps=1e-6,
        masked_image_modeling=True,
        patch_drop_min_patches=-1,
        patch_drop_max_patches=-1,
        patch_drop_at_eval=False,
        add_pos_same_dtype=False,
        patch_dropping=True,
        post_encoder_params=None,
        decoder=partial(
            Decoder,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=4,
            decoder_embed_dim=384,
            embed_dim=768,
            learnable_pos_embed=False,
            qkv_bias=True,
        ),
        mask_token_embed_dim=None,
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=384, out_features=1536),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def vit_base_mae_finetune_ssv2(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=12,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=768,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
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
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=768, out_features=174),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model



def vit_base_mae_finetune_in1k(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=12,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=768,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
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
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=768, out_features=1000),
        init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model



def vit_large_mae_pretraining(ckpt_path=None):
    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=1024,
        depth=24,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=16,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=1024,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
        layer_norm_eps=1e-6,
        masked_image_modeling=True,
        patch_drop_min_patches=-1,
        patch_drop_max_patches=-1,
        patch_drop_at_eval=False,
        add_pos_same_dtype=False,
        patch_dropping=True,
        post_encoder_params=None,
        decoder=partial(
            Decoder,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=4,
            decoder_embed_dim=512,
            embed_dim=1024,
            learnable_pos_embed=False,
            qkv_bias=True,
        ),
        mask_token_embed_dim=None,
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=512, out_features=1536),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def vit_large_mae_finetune_ssv2(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=1024,
        depth=24,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=16,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=1024,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
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
    )

    # NOTE: Head config for this model has funcky dropout in head
    # ckpt loeading is different.
    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=1024, out_features=174),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.01),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model



def vit_large_mae_finetune_in1k(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=1024,
        depth=24,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=16,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=1024,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
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
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=1024, out_features=1000),
        init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def vit_huge_mae_pretraining(ckpt_path=None):
    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 14, 14],
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=16,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 14, 14],
                    out_channels=1280,
                    stride=[2, 14, 14],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
        layer_norm_eps=1e-6,
        masked_image_modeling=True,
        patch_drop_min_patches=-1,
        patch_drop_max_patches=-1,
        patch_drop_at_eval=False,
        add_pos_same_dtype=False,
        patch_dropping=True,
        post_encoder_params=None,
        decoder=partial(
            Decoder,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=8,
            decoder_embed_dim=512,
            embed_dim=1280,
            learnable_pos_embed=False,
            qkv_bias=True,
        ),
        mask_token_embed_dim=None,
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=512, out_features=1176),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def vit_huge_mae_finetune_ssv2(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 14, 14],
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=16,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 14, 14],
                    out_channels=1280,
                    stride=[2, 14, 14],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
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
    )

    # NOTE: Head config for this model has funcky dropout in head
    # ckpt loeading is different.
    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=1280, out_features=174),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.01),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def vit_huge_mae_finetune_in1k(ckpt_path=None):

    trunk = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 14, 14],
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=16,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        force_cast_ln_fp32=False,
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        non_skip_wt=1.0,
        non_skip_wt_learnable=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 14, 14],
                    out_channels=1280,
                    stride=[2, 14, 14],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
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
    )

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=1280, out_features=1000),
        init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05),
    )

    model = OmniMAE(trunk, head) 
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model