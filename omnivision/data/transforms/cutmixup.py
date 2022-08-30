# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0, with modifications by Matthew Leavitt
(ito@fb.com; matthew.l.leavitt@gmail.com). Modifications are described here and
notated where present in the code.

Modifications:
- _mix_batch.__call__() now checks device of data its passed, and passes
device argument accordingly. Previous behavior allowed called functions to
default to using cuda, which caused an error when using CPU-based data.

COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import collections.abc as abc
import logging
from typing import Any, Callable

import numpy as np
import torch
from omnivision.data.api import VisionSample
from omnivision.utils.generic import convert_to_one_hot


class CutMixUp(Callable):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, batch):
        """
        This collator implements CutMix (https://arxiv.org/abs/1905.04899) and/or
        MixUp (https://arxiv.org/abs/1710.09412) via ClassyVision's
        implementation (link when publicly available).

        kwargs:
        :mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        :cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        :cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active
        and uses this vs alpha if not None.
        :prob (float): probability of applying mixup or cutmix per batch or element
        :switch_prob (float): probability of switching to cutmix instead of mixup
        when both are active
        :mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of
        elements), 'elem' (element)
        :correct_lam (bool): apply lambda correction when cutmix bbox clipped by
        image borders
        :label_smoothing (float): apply label smoothing to the mixed target tensor
        :num_classes (int): number of classes for target


        The collators collates the batch for the following input (assuming k-copies of image):

        Input:
            batch: Example
                    batch = [
                        {"data" : [img1_0, ..., img1_k], ..},
                        {"data" : [img2_0, ..., img2_k], ...},
                        ...
                    ]

        Returns: Example output:
                    output = {
                                "data": torch.tensor([img1_0, ..., imgN_0],
                                    [img1_k, ..., imgN_k]) ..
                            }
        """
        assert isinstance(batch, VisionSample)

        # Instantiate CutMix + Mixup (CutMixUp!) object
        cutmixup_transform_obj = Mixup(**self.kwargs)

        # Get data and labels into format accepted by Mixup
        cutmixup_output = cutmixup_transform_obj(
            {"input": batch.vision, "target": batch.label}
        )
        batch.vision = cutmixup_output["input"]
        batch.label = cutmixup_output["target"]

        return batch


# Modification/addition
def data_back_to_input_form(data, labels, data_valid, data_idx):
    """
    "De"-collates data back into their form when originally passed.
    """
    assert len(data) == len(labels)
    assert len(data_idx) == len(data_valid)
    data_input_form = []
    num_duplicates, num_images = len(data), len(data[0])
    for sample_i in range(num_images):
        sample_input_form = {"data": [], "data_valid": [], "data_idx": [], "label": []}
        for duplicate_i in range(num_duplicates):
            valid_and_idx_i = sample_i + (num_duplicates * duplicate_i)
            sample_input_form["data"].append(data[duplicate_i][sample_i])
            sample_input_form["label"].append(labels[duplicate_i][sample_i].tolist())
            sample_input_form["data_idx"].append(data_idx[valid_and_idx_i].item())
            sample_input_form["data_valid"].append(data_valid[valid_and_idx_i].item())
        data_input_form.append(sample_input_form)
    return data_input_form


def _recursive_mixup(sample: Any, permuted_indices: torch.Tensor, coeff: float):
    if isinstance(sample, (tuple, list)):
        mixed_sample = []
        for s in sample:
            mixed_sample.append(_recursive_mixup(s, permuted_indices, coeff))

        return mixed_sample if isinstance(sample, list) else tuple(mixed_sample)
    elif isinstance(sample, abc.Mapping):
        mixed_sample = {}
        for key, val in sample.items():
            mixed_sample[key] = _recursive_mixup(val, permuted_indices, coeff)

        return mixed_sample
    else:
        assert torch.is_tensor(sample), "sample is expected to be a pytorch tensor"
        # Assume training data is at least 3D tensor (i.e. 1D data). We only
        # mixup content data tensor (e.g. video clip), and skip
        # other tensors, such as frame_idx and timestamp in video clip samples.
        if sample.ndim >= 3:
            sample = coeff * sample + (1.0 - coeff) * sample[permuted_indices, :]

        return sample


def one_hot(x, num_classes, on_value=1.0, off_value=0.0, device="cuda"):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(
        1, x, on_value
    )


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0, device="cuda"):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(
        target, num_classes, on_value=on_value, off_value=off_value, device=device
    )
    y2 = one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
        device=device,
    )
    return y1 * lam + y2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin
            (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9
    range for max.
    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image
        size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(
        int(img_h * minmax[0]), int(img_h * minmax[1]), size=count
    )
    cut_w = np.random.randint(
        int(img_w * minmax[0]), int(img_w * minmax[1]), size=count
    )
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(
    img_shape, lam, ratio_minmax=None, correct_lam=True, count=None
):
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """Mixup/Cutmix that applies different params to each element or whole batch
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is
        active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or
        element
        switch_prob (float): probability of switching to cutmix instead of
        mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair
        of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by
        image borders
        label_smoothing (float): apply label smoothing to the mixed target
        tensor
        num_classes (int): number of classes for target
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = (
            correct_lam  # correct lambda based on clipped area for cutmix
        )
        self.mixup_enabled = (
            True  # set to false to disable mixing (intended tp be set by train loop)
        )

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(
                        self.cutmix_alpha, self.cutmix_alpha, size=batch_size
                    ),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size),
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(
                    self.mixup_alpha, self.mixup_alpha, size=batch_size
                )
            elif self.cutmix_alpha > 0.0:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(
                    self.cutmix_alpha, self.cutmix_alpha, size=batch_size
                )
            else:
                assert AssertionError, (
                    "One of mixup_alpha > 0., cutmix_alpha > 0.,"
                    "cutmix_minmax not None should be true."
                )
            lam = np.where(
                np.random.rand(batch_size) < self.mix_prob,
                lam_mix.astype(np.float32),
                lam,
            )
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.0
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert AssertionError, (
                    "One of mixup_alpha > 0., cutmix_alpha > 0.,"
                    "cutmix_minmax not None should be true."
                )
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam,
            )
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, sample):
        x = sample["input"]
        orig_shape = x.shape
        if len(orig_shape) == 5:
            # Video case, convert to image by moving time to channels
            x = x.flatten(1, 2)
        target = sample["target"]
        len_to_mix = len(x)
        if len(x) % 2 != 0:
            len_to_mix = len_to_mix - 1
            logging.warning(
                "Batch size should be even when using this. "
                "Got %d. Applying cutmix to first %d elements.",
                len(x),
                len_to_mix,
            )
        # rgirdhar: using torch.narrow to make sure it is a view and shares
        # the same storage (since the function modifies it inplace)
        if self.mode == "elem":
            lam = self._mix_elem(torch.narrow(x, 0, 0, len_to_mix))
        elif self.mode == "pair":
            lam = self._mix_pair(torch.narrow(x, 0, 0, len_to_mix))
        else:
            lam = self._mix_batch(torch.narrow(x, 0, 0, len_to_mix))
        # Modified to pass device argument based on target.device to prevent
        # failure on CPU-based data.
        target_1hot = convert_to_one_hot(target.view(-1, 1), self.num_classes).to(
            torch.float
        )
        target_1hot[:len_to_mix] = mixup_target(
            target[:len_to_mix],
            self.num_classes,
            lam,
            self.label_smoothing,
            device=target.device,
        )
        if len(orig_shape) == 5:
            # Video case, convert back to video
            x = x.view(orig_shape)
        return {"input": x, "target": target_1hot}
