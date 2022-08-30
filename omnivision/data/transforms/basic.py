# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
import torch.nn as nn
from omnivision.data.transforms.pytorchvideo import uniform_crop
from PIL import Image
from torchvision import transforms


class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = tuple(ordering)

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)


class PILToRGB(nn.Module):
    """
    PIL Image to RGB
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: Image) -> Image:
        return image.convert("RGB")


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well. It's useful for 3x4 testing (eg in SwinT)
        or 3x10 testing in SlowFast etc.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 6:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = [0, 1, 2]
        elif num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError(
                "Nothing else supported yet, "
                "slowfast only takes 0, 1, 2 as arguments"
            )

    def forward(self, video: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert video.ndim == 4, "Must be (C,T,H,W)"
        res = []
        for spatial_idx in self.crops_to_ext:
            res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
        if not self.flipped_crops_to_ext:
            return res
        flipped_video = transforms.functional.hflip(video)
        for spatial_idx in self.flipped_crops_to_ext:
            res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res
