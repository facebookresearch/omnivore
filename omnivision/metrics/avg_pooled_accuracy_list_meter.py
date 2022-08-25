# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch

from .accuracy import Accuracy

# modified from vissl/meters/accuracy_list_meter_avg_pooled.py
class AvgPooledAccuracyListMeter(Accuracy):
    def update(
        self,
        model_output: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ):
        if isinstance(model_output, list):
            assert len(model_output) == 1
            model_output = model_output[0]
        assert isinstance(model_output, torch.Tensor)
        model_output_reshaped = model_output.reshape(
            (-1, target.size(0)) + model_output.shape[1:]
        )

        model_output_avg = torch.mean(model_output_reshaped, dim=0)
        super().update(model_output_avg, target)
