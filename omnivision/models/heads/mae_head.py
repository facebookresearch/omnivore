# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from timm.models.layers import trunc_normal_


class MAEHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.projector = nn.Linear(in_features, out_features, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        if isinstance(batch, tuple):
            batch = batch[1]
            return self.projector(batch)
