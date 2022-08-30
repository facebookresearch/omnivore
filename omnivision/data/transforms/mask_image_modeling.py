# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Parts of code are modified from https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py

import math
import random
from abc import ABC
from typing import List, Tuple

import numpy as np
import torch


def get_image_dims(img):
    # If an image, convert to singleton video
    if img.ndim == 3:
        time_dim = 1
        squeeze_dim = True
    else:
        time_dim = img.shape[-3]
        squeeze_dim = False
    return squeeze_dim, time_dim, img.shape[-2], img.shape[-1]


def get_pred_ratio(pred_ratio, pred_ratio_var):
    if isinstance(pred_ratio, list):
        curr_pred_ratio = []
        for prm, prv in zip(pred_ratio, pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            curr_pred_ratio.append(pr)
        curr_pred_ratio = random.choice(curr_pred_ratio)
    else:
        assert pred_ratio >= pred_ratio_var
        curr_pred_ratio = (
            random.uniform(
                pred_ratio - pred_ratio_var,
                pred_ratio + pred_ratio_var,
            )
            if pred_ratio_var > 0
            else pred_ratio
        )
    return curr_pred_ratio


class Masking(ABC):
    pass


class BlockMasking(Masking):
    def __init__(self, pred_aspect_ratio: Tuple[float] = (0.3, 1 / 0.3)):
        self.pred_aspect_ratio = pred_aspect_ratio

    def __call__(self, T: int, H: int, W: int, high: float) -> np.ndarray:
        assert T == 1, "Does not support videos yet"
        mask = np.zeros((T, H, W), dtype=bool)
        mask_count = 0
        log_aspect_ratio = tuple(map(lambda x: math.log(x), self.pred_aspect_ratio))
        while mask_count < high:
            max_mask_patches = high - mask_count

            delta = 0
            for _ in range(10):
                low = (min(H, W) // 3) ** 2
                target_area = random.uniform(low, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < W and h < H:
                    top = random.randint(0, H - h)
                    left = random.randint(0, W - w)
                    num_masked = mask[top : top + h, left : left + w].sum()
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[0, i, j] == 0:
                                    mask[0, i, j] = 1
                                    delta += 1

                if delta > 0:
                    break

            if delta == 0:
                break
            else:
                mask_count += delta
        return mask


class RandMasking(Masking):
    @staticmethod
    def __call__(
        T: int, H: int, W: int, high: float, shuffle: bool = True
    ) -> np.ndarray:
        mask = np.hstack(
            [
                np.zeros(T * H * W - int(high)),
                np.ones(int(high)),
            ]
        ).astype(bool)
        if shuffle:
            np.random.shuffle(mask)
        mask = mask.reshape(T, H, W)
        return mask


class TubeMasking(Masking):
    """
    Extends any frame mask to the temporal extent of a video.
    """

    def __init__(self, frame_masking: Masking):
        self.frame_masking = frame_masking

    def __call__(self, T: int, H: int, W: int, high: float):
        # Get a frame level mask with 1/T the masking
        frame_mask = self.frame_masking(1, H, W, high / T)
        # replicate the frame mask for all the T frames
        return frame_mask.repeat(T, 0)


class CausalMasking(RandMasking):
    """
    Masks out the first N tokens in a raster order.
    """

    @classmethod
    def __call__(cls, *args, **kwargs) -> np.ndarray:
        kwargs["shuffle"] = False
        T = super().__call__(*args, **kwargs)
        return T


class RandomFrameMasking(Masking):
    """
    Masks out random frames from the clip.
    """

    @staticmethod
    def __call__(T: int, H: int, W: int, high: float) -> np.ndarray:
        mask = np.zeros((T, H, W), dtype=np.bool)
        # Can't be exact to high since need to mask the full frame, but round it
        # high is the max number of tokens to mask in the input
        pred_ratio = high * 1.0 / (T * H * W)
        frame_ids = np.arange(T, dtype=int)
        np.random.shuffle(frame_ids)
        frame_ids_to_mask = frame_ids[: math.floor(T * pred_ratio)]
        mask[frame_ids_to_mask, ...] = True
        return mask


def ibot_style_mask_image(
    image: torch.Tensor,
    patch_size: List[int],  # [patch_t, patch_h, patch_w]
    pred_ratio: Tuple[float],
    pred_ratio_var: Tuple[float],
    pred_shape: Masking = RandMasking,
    precomputed_pred_ratio: float = None,
):
    squeeze_dim, img_t, img_h, img_w = get_image_dims(image)
    T = max(img_t // patch_size[0], 1)
    H = img_h // patch_size[1]
    W = img_w // patch_size[2]
    if precomputed_pred_ratio is None:
        precomputed_pred_ratio = get_pred_ratio(
            pred_ratio=pred_ratio, pred_ratio_var=pred_ratio_var
        )
    # high is the max number of tokens to mask in the input
    high = precomputed_pred_ratio * T * H * W
    mask = pred_shape(T, H, W, high)

    if squeeze_dim:
        # Remove the time dim from the mask since the image doesn't have it
        mask = np.squeeze(mask, axis=0)
    ret_dict = {
        "data": image,
        "mask": torch.from_numpy(mask),
    }
    return ret_dict


class MaskImageModeling:
    def __init__(
        self,
        pred_ratio: Tuple[float],
        pred_ratio_var: Tuple[float],
        patch_size: List[int],
        pred_shape: Masking = RandMasking,
        mim_start_epochs: float = 0,
    ):
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.patch_size = patch_size
        self.pred_shape = pred_shape
        self.mim_start_epochs = int(math.floor(mim_start_epochs))
        if self.mim_start_epochs != mim_start_epochs:
            raise NotImplementedError(
                "The mim_start_epochs must be integer. Fractional not supported yet."
            )
        self.current_epoch = None

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __call__(self, image, do_not_mask=False):
        precomputed_pred_ratio = None
        if self.mim_start_epochs > 0 and self.mim_start_epochs > self.current_epoch:
            precomputed_pred_ratio = 0
        if do_not_mask is True:
            precomputed_pred_ratio = 0
        return ibot_style_mask_image(
            image,
            patch_size=self.patch_size,
            pred_ratio=self.pred_ratio,
            pred_ratio_var=self.pred_ratio_var,
            pred_shape=self.pred_shape,
            precomputed_pred_ratio=precomputed_pred_ratio,
        )
