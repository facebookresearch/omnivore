# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch.nn as nn


def make_conv_or_linear(layer, init_weight=None, init_bias=None):
    layer_params = hydra.utils.instantiate(layer, _convert_="all")
    if init_weight is not None:
        hydra.utils.instantiate(
            init_weight, _convert_="all", tensor=layer_params.weight.data
        )
    if init_bias is not None:
        hydra.utils.instantiate(
            init_bias, _convert_="all", tensor=layer_params.bias.data
        )
    return layer_params


def reshape_and_init_as_mlp(tensor):
    # Based on MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
    nn.init.xavier_uniform_(tensor.view([tensor.shape[0], -1]))


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


class PadIm2Video(Im2Video):
    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__(time_dim=time_dim)
        assert ntimes > 0
        assert pad_type in ["zero", "repeat"]
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x):
        x = super().forward(x)
        if x.shape[self.time_dim] == 1:
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
            elif self.pad_type == "zero":
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim]
                x = nn.functional.pad(x, padarg)
        return x
