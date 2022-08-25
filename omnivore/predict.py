#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Download the weights in ./checkpoints and ImageNet 1K ID to class mappings beforehand
wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
wget https://dl.fbaipublicfiles.com/omnivore/sunrgbd_classnames.json
"""
import json
from pathlib import Path

import cog

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.omnivore_model import get_all_heads, OmnivoreModel
from models.swin_transformer_3d import SwinTransformer3D
from PIL import Image
from transforms import DepthNorm

CHECKPOINT_PATHS = {
    "omnivore_swinT": "checkpoints/swinT_checkpoint.torch",
    "omnivore_swinS": "checkpoints/swinS_checkpoint.torch",
    "omnivore_swinB": "checkpoints/swinB_checkpoint.torch",
    "omnivore_swinB_in21k": "checkpoints/swinB_In21k_checkpoint.torch",
    "omnivore_swinL_in21k": "checkpoints/swinL_In21k_checkpoint.torch",
}


class Predictor(cog.Predictor):
    def setup(self):
        self.device = "cuda:0"
        self.models = {
            "omnivore_swinT": omnivore_swinT(),
            "omnivore_swinS": omnivore_swinS(),
            "omnivore_swinB": omnivore_swinB(),
            "omnivore_swinB_in21k": omnivore_swinB(
                checkpoint_name="omnivore_swinB_in21k"
            ),
            "omnivore_swinL_in21k": omnivore_swinL(
                checkpoint_name="omnivore_swinL_in21k"
            ),
        }

        with open("imagenet_class_index.json", "r") as f:
            imagenet_classnames = json.load(f)

        # Create an id to label name mapping
        self.imagenet_id_to_classname = {}
        for k, v in imagenet_classnames.items():
            self.imagenet_id_to_classname[k] = v[1]

        with open("sunrgbd_classnames.json", "r") as f:
            self.sunrgbd_id_to_classname = json.load(f)

        self.image_transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.rgbd_transform = T.Compose(
            [
                DepthNorm(max_depth=75.0, clamp_max_before_scale=True),
                T.Resize(224),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.0418],
                    std=[0.229, 0.224, 0.225, 0.0295],
                ),
            ]
        )

    @cog.input(
        "image",
        type=Path,
        help="input image",
    )
    @cog.input(
        "model_name",
        type=str,
        default="omnivore_swinB",
        options=[
            "omnivore_swinB",
            "omnivore_swinT",
            "omnivore_swinS",
            "omnivore_swinB_in21k",
            "omnivore_swinL_in21k",
        ],
        help="Choose a model",
    )
    @cog.input(
        "topk",
        type=int,
        min=1,
        max=10,
        default=5,
        help="Choose top k predictions to return.",
    )
    def predict(self, image, model_name, topk):
        model = self.models[model_name]
        model.to(self.device)
        model.eval()

        image = Image.open(str(image)).convert("RGB")

        # Inference with Images
        image = self.image_transform(image)

        # The model expects inputs of shape: B x C x T x H x W
        image = image[None, :, None, ...]
        image = image.to(self.device)
        prediction = model(image, input_type="image")
        prediction = F.softmax(prediction, dim=1)
        pred_classes = prediction.topk(k=5).indices

        pred_class_names = [
            self.imagenet_id_to_classname[str(i.item())] for i in pred_classes[0]
        ]
        return f"Top {topk} predicted labels: %s" % ", ".join(pred_class_names)


def omnivore_base(trunk, head_dim_in=1024, checkpoint_name="omnivore_swinB"):

    heads = get_all_heads(dim_in=head_dim_in)

    path = CHECKPOINT_PATHS[checkpoint_name]
    # All models are loaded onto CPU by default
    checkpoint = torch.load(path, map_location="cpu")
    trunk.load_state_dict(checkpoint["trunk"])

    heads.load_state_dict(checkpoint["heads"])

    return OmnivoreModel(trunk=trunk, heads=heads)


def omnivore_swinB(checkpoint_name="omnivore_swinB"):

    trunk = SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(16, 7, 7),
        drop_path_rate=0.3,  # TODO: set this based on the final models
        patch_norm=True,  # Make this the default value?
        depth_mode="summed_rgb_d_tokens",  # Make this the default value?
    )

    return omnivore_base(
        trunk=trunk,
        head_dim_in=1024,  # embed_dim * 8 = 128*8
        checkpoint_name=checkpoint_name,
    )


def omnivore_swinS():
    trunk = SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
    )

    return omnivore_base(
        trunk=trunk,
        head_dim_in=768,  # 96*8
        checkpoint_name="omnivore_swinS",
    )


def omnivore_swinT():

    trunk = SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        drop_path_rate=0.2,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
    )

    return omnivore_base(
        trunk=trunk,
        head_dim_in=768,  # 96*8
        checkpoint_name="omnivore_swinT",
    )


def omnivore_swinL(checkpoint_name=""):

    assert checkpoint_name != "", "checkpoint_name must be provided"

    trunk = SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
    )

    return omnivore_base(
        trunk=trunk,
        head_dim_in=1536,  # 192*8
        checkpoint_name=checkpoint_name,
    )
