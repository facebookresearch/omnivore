# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class ScaledLoss(nn.Module):
    def __init__(self, loss_fn, scale):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale = scale

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs) * self.scale
