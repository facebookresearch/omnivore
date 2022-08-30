# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
import torch.nn as nn
from omnivision.data.api import VisionMaskSample
from omnivision.losses import BaseLoss


class MAELoss(BaseLoss):
    def __init__(
        self,
        patch_size: int = 16,
        norm_pix_loss: bool = True,
        norm_pix_per_channel: bool = False,
        unnormalize_img: Any = None,
        pad_object: Optional[nn.Module] = None,
    ) -> float:
        """
        MAE loss implementation that computes a simple MSE loss with optional patch wise
        normalization.

        Args:
            patch_size (int, optional): Defaults to 16.
            norm_pix_loss (bool, optional): Normalize the pixel values using the mean and std of all
             the pixel values within a patch. Defaults to True.
            norm_pix_per_channel (bool, optional): Normalize the pixel values within a patch
             separately for each of the RGB channels. Defaults to False.
            unnormalize_img (Any, optional): Defaults to None.
            pad_object ([type], optional): Defaults to None.
        """
        super().__init__()
        self.patch_size = patch_size
        if isinstance(self.patch_size, int):
            # Must be a tuple specifying the full patch size so it works with videos
            self.patch_size = [self.patch_size, self.patch_size]
        self.norm_pix_loss = norm_pix_loss
        # Computes the pix norm per channel (what VideoMAE does)
        # instead of altogether for all channels (what ImageMAE does)
        self.norm_pix_per_channel = norm_pix_per_channel
        assert (
            not self.norm_pix_per_channel or self.norm_pix_loss
        ), "Must specify self.norm_pix_loss if using norm_pix_per_channel"
        self.unnormalize_img = unnormalize_img

        # Use this to process the image before patchifying it, eg padding the
        # image by replicate. It can just interpolate
        # from the MODEL's patchify layer
        self.pad_object = pad_object

    def compute_mae_loss(self, pred, mask, img):
        mask = mask.reshape(mask.shape[0], -1)
        if self.pad_object is not None:
            img = self.pad_object(img)
        # Based on
        # https://github.com/MCG-NJU/VideoMAE/blob/a8dd8eedf955b3e3cc86c701e19e2553b4665154/engine_for_pretraining.py#L37-L59
        if self.unnormalize_img is not None:
            img_mean = (
                torch.as_tensor(self.unnormalize_img[0])
                .to(img.device)
                .reshape([1, -1] + [1] * (img.ndim - 2))
            )
            img_std = (
                torch.as_tensor(self.unnormalize_img[1])
                .to(img.device)
                .reshape([1, -1] + [1] * (img.ndim - 2))
            )
            img = img * img_std + img_mean
        target = self.patchify(img)
        patches_dim = -2
        if self.norm_pix_loss:
            if not self.norm_pix_per_channel:
                # Merge the channel with patches and compute mean
                # over all channels of all patches.
                # Else, will compute a mean for each channel separately
                target = torch.flatten(target, patches_dim)
                patches_dim = -1
            mean = target.mean(dim=patches_dim, keepdim=True)
            var = target.var(dim=patches_dim, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            if self.norm_pix_per_channel:
                # In this case we didn't flatten channel dim earlier, so flatten now
                target = torch.flatten(target, -2)
        else:
            target = torch.flatten(target, patches_dim)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mask_sum = mask.sum().clamp(min=1)
        loss = (loss * mask).sum() / mask_sum

        return loss

    def core_forward(self, output: torch.Tensor, sample: VisionMaskSample):
        """
        Modified from: https://github.com/facebookresearch/mae/blob/main/models_mae.py
        """
        img = sample.vision
        mask = sample.mask
        pred = output
        return self.compute_mae_loss(pred, mask, img)

    def patchify(self, imgs):
        """
        Modified from: https://github.com/facebookresearch/mae/blob/main/models_mae.py
        """
        assert imgs.shape[-2] == imgs.shape[-1]  # Spatial dimensions match up
        p = self.patch_size

        # Add a dummy time dimension to 2D patches for consistency.
        # Since it is 1, it will not affect the final number of patches
        if len(p) == 2:
            p = [
                1,
            ] + p
            imgs = imgs.unsqueeze(-3)
        assert imgs.ndim - 2 == len(p)  # except batch and channel dims
        for i in range(1, len(p) + 1):
            assert (
                imgs.shape[-i] % p[-i] == 0
            ), f"image shape {imgs.shape} & patch shape {p} mismatch at index {i}"

        h = imgs.shape[-1] // p[-1]
        w = imgs.shape[-2] // p[-2]
        t = imgs.shape[-3] // p[-3]
        x = imgs.reshape(shape=(imgs.shape[0], 3, t, p[-3], h, p[-2], w, p[-1]))
        x = torch.einsum("nctphqwr->nthwpqrc", x)
        x = x.reshape(shape=(imgs.shape[0], t * h * w, p[-3] * p[-2] * p[-1], 3))

        return x
