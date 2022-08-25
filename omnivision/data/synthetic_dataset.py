# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from omnivision.data.api import VisionSample
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, tensor_shape, length, label=0, value=1) -> None:
        self.tensor = torch.full(tuple(tensor_shape), float(value))
        self.label = label
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> torch.Tensor:
        return VisionSample(
            vision=self.tensor,
            label=self.label,
            data_idx=idx,
            data_valid=True,
        )
