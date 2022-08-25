# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code modified from,
# a) https://github.com/facebookresearch/SlowFast
# b) https://github.com/facebookresearch/vissl

import math

from typing import Sequence, Union

import numpy as np
import pytorchvideo
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class UniformCrop(nn.Module):
    """
    Wrapper around pytorchvideo.transforms.functional.uniform_crop, without
        needing the dictionary interface required in
        pytorchvideo.transforms.UniformCropVideo
    """

    def __init__(self, size: int = 256, spatial_idx: Union[int, Sequence[int]] = 1):
        super().__init__()
        self.size = size
        # Convert to list to deal with multiple crops
        if isinstance(spatial_idx, int):
            self.spatial_idx = [spatial_idx]
        else:
            self.spatial_idx = spatial_idx

    def forward(self, video):
        res = []
        for i in self.spatial_idx:
            res.append(
                pytorchvideo.transforms.functional.uniform_crop(video, self.size, i)
            )
        if len(res) == 1:  # Don't return as list if only 1 spatial index
            res = res[0]
        return res


class VideoToListOfFrames(nn.Module):
    def forward(self, video):
        """
        Converts a video tensor to image list by moving the time dimension to
            batch dimension. It will work with the average pooled accuracy
            meters.
        Args:
            video (C, T, H, W)
        Returns:
            list of [(C, H, W)] * T
        """
        assert video.ndim == 4
        return [el.squeeze(1) for el in torch.split(video, 1, dim=1)]


class TemporalCrop(nn.Module):
    """
    Convert the video into smaller clips temporally.
    """

    def __init__(
        self, frames_per_clip: int = 8, stride: int = 8, frame_stride: int = 1
    ):
        super().__init__()
        self.frames = frames_per_clip
        self.stride = stride
        self.frame_stride = frame_stride

    def forward(self, video):
        assert video.ndim == 4, "Must be (C, T, H, W)"
        res = []
        for start in range(
            0, video.size(1) - (self.frames * self.frame_stride) + 1, self.stride
        ):
            end = start + (self.frames) * self.frame_stride
            res.append(video[:, start : end : self.frame_stride, ...])
        return res


class TemporalCrop2(nn.Module):
    """
    Convert the video into smaller clips temporally. This version is inspired
        from Swin Transformer Video. It crops central clips as opposed to
        starting from the first frame. Also, takes as input the desired
        number of clips instead of the
        https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/mmaction/datasets/pipelines/loading.py#L164
    """

    def __init__(self, frames_per_clip: int = 8, num_clips: int = 5):
        super().__init__()
        self.frames = frames_per_clip
        self.num_clips = num_clips

    def forward(self, video):
        assert video.ndim == 4, "Must be (C, T, H, W)"
        num_frames = video.size(1)
        avg_interval = (num_frames - self.frames) / self.num_clips
        base_offsets = np.arange(self.num_clips) * avg_interval
        clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        res = []
        for start in clip_offsets:
            res.append(video[:, start : start + self.frames, ...])
        return res


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

    def forward(self, videos: Sequence[torch.Tensor]):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


class TemporalSegmentSampler(nn.Module):
    """
    Samples frames from video by dividing into segments and sampling a frame
    from each segment similar to TSN (ECCV'16).
    Also useful for SS-v2 training/testing. Modified from
    https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
    """

    def __init__(self, num_samples: int, train_mode: bool = False):
        super().__init__()
        self.num_segments = num_samples
        self.train_mode = train_mode

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Video tensor with shape (C, T, H, W)
        """
        num_frames = x.size(1)
        seg_size = float(num_frames - 1) / self.num_segments
        seq = []
        for i in range(self.num_segments):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.train_mode:
                seq.append(np.random.randint(low=start, high=(end + 1)))
            else:
                seq.append((start + end) // 2)
        return x[:, seq, :, :]


class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)


class Replicate(nn.Module):
    """
    Replicate a tensor N times and return a list of N copies
    N copies do *not* share storage
    """

    def __init__(self, num_times):
        super().__init__()
        self.num_times = num_times

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        ret_list = [x.clone() for _ in range(self.num_times)]
        return ret_list

    def extra_repr(self):
        return f"num_times={self.num_times}"


class ShuffleList(nn.Module):
    """
    Shuffle a list of objects randomly.
    Useful when shuffling temporal crops.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert isinstance(x, list), f"Expected list input. Found {type(x)}"
        inds = torch.randperm(len(x)).tolist()
        new_x = [x[ind] for ind in inds]
        return new_x


class SingletonVideoToImgPil(nn.Module):
    """
    Convert a 1 frame video to PIL image for image transforms.
    """

    def forward(self, x):
        assert x.size(1) == 1, x.shape
        x = x[:, 0, ...]
        if x.dtype == torch.float:
            x = (x * 255.0).to(torch.uint8)
        return Image.fromarray(x.cpu().permute(1, 2, 0).numpy())
