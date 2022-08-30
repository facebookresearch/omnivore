# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn
from omnivision.data.api import Sample


@dataclass
class LossWithUpdatedOutput:
    loss: torch.Tensor
    output: torch.Tensor


def wrap_base_loss(loss):
    if isinstance(loss, BaseLoss):
        return loss
    return BaseLoss(core_loss=loss)


class BaseLoss(nn.Module):
    """
    The base Omnivore loss API.
    By default all losses get wrapped into this loss.
    """

    def __init__(self, *args, core_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_loss = core_loss

    def core_forward(self, output: torch.Tensor, sample: Sample, *args, **kwargs):
        return self.core_loss(output, sample.label, *args, **kwargs)

    def forward(self, output: torch.Tensor, sample: Sample, *args, **kwargs):
        loss_out = self.core_forward(output, sample, *args, **kwargs)
        if isinstance(loss_out, LossWithUpdatedOutput):
            loss, output = loss_out.loss, loss_out.output
        else:
            loss = loss_out
        return loss, output
