#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial

import torch
import torch.nn as nn

from omnivision.models.vision_transformer import (
    Attention,
    Decoder,
    PadIm2Video,
    VisionTransformer,
)

from timm.models.layers import trunc_normal_
from torch.hub import load_state_dict_from_url


CHECKPOINT_PATHS = {
    "omnimae_vitB_pretrain": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitb_pretrain.torch",
    "omnimae_vitB_ft_in1k": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitb_in1k_ft.torch",
    "omnimae_vitB_ft_ssv2": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitb_ssv2_ft.torch",
    "omnimae_vitL_pretrain": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitl_pretrain.torch",
    "omnimae_vitL_ft_in1k": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitl_in1k_ft.torch",
    "omnimae_vitL_ft_ssv2": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitl_ssv2_ft.torch",
    "omnimae_vitH_pretrain": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vith_pretrain.torch",
    "omnimae_vitH_ft_in1k": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vith_in1k_ft.torch",
    "omnimae_vitH_ft_ssv2": "https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vith_ssv2_ft.torch",
}


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
        return self.head(outputs)
  

def _load_checkpoint(model, checkpoint_name, pretrained, progress=True):
    if pretrained:
        path = CHECKPOINT_PATHS[checkpoint_name]
        print(f"Loading {checkpoint_name} from {path}")
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
        assert len(missing_keys) == 0 and len(unexpected_keys) == 0
    return model


def vit_base_mae_pretraining(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitB_pretrain"
    )
    return model


def vit_base_mae_finetune_ssv2(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitB_ft_ssv2"
    )
    return model


def vit_base_mae_finetune_in1k(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitB_ft_in1k"
    )
    return model


def vit_large_mae_pretraining(pretrained=True):
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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitL_pretrain"
    )
    return model


def vit_large_mae_finetune_ssv2(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitL_ft_ssv2"
    )
    return model


def vit_large_mae_finetune_in1k(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitL_ft_in1k"
    )
    return model


def vit_huge_mae_pretraining(pretrained=True):
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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitH_pretrain"
    )
    return model


def vit_huge_mae_finetune_ssv2(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitH_ft_ssv2"
    )
    return model


def vit_huge_mae_finetune_in1k(pretrained=True):

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
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
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
        patch_drop_max_patches=-1,
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
    model = _load_checkpoint(
        model=model, pretrained=pretrained, checkpoint_name="omnimae_vitH_ft_in1k"
    )
    return model
