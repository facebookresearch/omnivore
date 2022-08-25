# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from iopath.common.file_io import g_pathmgr
from omnivision.data.api import VisionSample
from omnivision.utils.data import (
    get_mean_image,
    IdentityTransform,
    SharedMemoryNumpyLoader,
)
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset


IDENTITY_TRANSFORM = IdentityTransform()
DEFAULT_SPATIAL_SIZE = 224


class PathDataset(Dataset, ABC):
    def __init__(
        self,
        path_file_list: List[str],
        label_file_list: List[str],
        remove_prefix="",
        new_prefix="",
        remove_suffix="",
        new_suffix="",
        transforms=None,
    ):
        """Creates a dataset where the metadata is stored in a numpy file.

        path_file_list: A list of paths which contain the path metadata file. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the metadata.
        label_file_list: A list of paths which contain the label metadata file. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the metadata.
        """
        self.is_initialized = False
        self.path_file_list = path_file_list
        self.label_file_list = label_file_list
        self.transforms = [] if transforms is None else transforms

        self.remove_prefix = remove_prefix
        self.new_prefix = new_prefix
        self.remove_suffix = remove_suffix
        self.new_suffix = new_suffix

        self.paths = None
        self.labels = None
        self.file_idx = None

        # used for shared memory
        self.label_sm_loader = SharedMemoryNumpyLoader()
        self.path_sm_loader = SharedMemoryNumpyLoader()

        self._load_data()
        self.num_samples = len(self.paths)
        assert len(self.paths) == len(
            self.labels
        ), f"Paths ({len(self.paths)}) != labels ({len(self.labels)})"
        logging.info(
            f"Created dataset from {self.path_file_list} of length: {self.num_samples}"
        )

    def _load_data(self):
        logging.info(f"Loading {self.label_file_list} with shared memory")
        self.labels, label_file_idx = self.label_sm_loader.load(self.label_file_list)
        logging.info(f"Loading {self.path_file_list} with shared memory")
        self.paths, path_file_idx = self.path_sm_loader.load(self.path_file_list)
        assert (
            label_file_idx == path_file_idx
        ), "Label file and path file were not found at the same index"
        self.is_initialized = True
        self.file_idx = path_file_idx

    def _replace_path_prefix(self, path, replace_prefix, new_prefix):
        if replace_prefix == "":
            path = new_prefix + path
        elif path.startswith(replace_prefix):
            return new_prefix + path[len(replace_prefix) :]
        else:
            raise ValueError(f"Cannot replace `{replace_prefix}`` prefix in `{path}`")
        return path

    def _replace_path_suffix(self, path, replace_suffix, new_suffix):
        if replace_suffix == "":
            path = path + new_suffix
        elif path.endswith(replace_suffix):
            return path[: -len(replace_suffix)] + new_suffix
        else:
            raise ValueError(f"Cannot replace `{replace_suffix}`` suffix in `{path}`")
        return path

    def __len__(self):
        return self.num_samples

    @abstractmethod
    def default_generator(self):
        pass

    @abstractmethod
    def load_object(self, path):
        pass

    def _get_path(self, idx):
        path = self._replace_path_prefix(
            self.paths[idx],
            replace_prefix=self.remove_prefix,
            new_prefix=self.new_prefix,
        )
        path = self._replace_path_suffix(
            path, replace_suffix=self.remove_suffix, new_suffix=self.new_suffix
        )
        return path

    def try_load_object(self, idx):
        is_success = True
        try:
            data = self.load_object(self._get_path(idx))
        except Exception:
            logging.warning(
                f"Couldn't load: {self.paths[idx]}. Exception:", exc_info=True
            )
            is_success = False
            data = self.default_generator()
        return data, is_success

    def get_label(self, idx):
        return None if self.labels is None else self.labels[idx]

    @staticmethod
    def create_sample(idx, data, label, is_success):
        return VisionSample(
            vision=data, label=int(label), data_idx=idx, data_valid=is_success
        )

    def apply_transforms(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, idx):
        data, is_success = self.try_load_object(idx)
        label = self.get_label(idx)
        sample = self.create_sample(idx, data, label, is_success)
        sample = self.apply_transforms(sample)
        return sample


class ImagePathDataset(PathDataset):
    def default_generator(self):
        return get_mean_image(DEFAULT_SPATIAL_SIZE)

    def load_object(self, path) -> Image:
        with g_pathmgr.open(path, "rb") as fopen:
            return Image.open(fopen).convert("RGB")


class ImageWithDepthPathDataset(ImagePathDataset):
    def __init__(
        self,
        depth_path_file_list: List[str],
        *args,
        remove_depth_prefix="",
        new_depth_prefix="",
        remove_depth_suffix="",
        new_depth_suffix="",
        **kwargs,
    ):
        """
        Shared Memory dataloader for RGB+Depth datasets.
        """
        super().__init__(*args, **kwargs)

        self.depth_path_file_list = depth_path_file_list

        self.remove_depth_prefix = remove_depth_prefix
        self.new_depth_prefix = new_depth_prefix
        self.remove_depth_suffix = remove_depth_suffix
        self.new_depth_suffix = new_depth_suffix

        self.depth_path_sm_loader = SharedMemoryNumpyLoader()

        logging.info(f"Loading {self.depth_path_file_list} with shared memory")
        self.depth_paths, depth_file_idx = self.depth_path_sm_loader.load(
            self.depth_path_file_list
        )

        assert (
            depth_file_idx == self.file_idx
        ), "Depth file and path file were not found at the same index"

    def _load_depth(self, image_path):
        """
        Returns:
            A (H, W, 1) tensor
        """
        with g_pathmgr.open(image_path, "rb") as fopen:
            # Depth is being saved as a .pt file instead
            # of as an image
            return torch.load(fopen)

    def _get_depth_path(self, idx):
        path = self._replace_path_prefix(
            self.depth_paths[idx],
            replace_prefix=self.remove_depth_prefix,
            new_prefix=self.new_depth_prefix,
        )
        path = self._replace_path_suffix(
            path,
            replace_suffix=self.remove_depth_suffix,
            new_suffix=self.new_depth_suffix,
        )
        return path

    def default_generator(self):
        image = get_mean_image(DEFAULT_SPATIAL_SIZE)
        depth = torch.zeros(
            (1, DEFAULT_SPATIAL_SIZE, DEFAULT_SPATIAL_SIZE), dtype=torch.float32
        )
        return torch.cat([tvf.to_tensor(image), depth], dim=0)

    def try_load_object(self, idx):
        image, is_success = super().try_load_object(idx)
        if is_success:
            try:
                depth = self._load_depth(self._get_depth_path(idx))
                if depth.ndim == 2:
                    depth = depth[None, ...]  # (1, H, W)
                image_with_depth = torch.cat(
                    [tvf.to_tensor(image), depth], dim=0
                )  # (4, H, W)
            except Exception:
                logging.warning(
                    f"Couldn't load depth image: {self.depth_paths[idx]}. Exception:",
                    exc_info=True,
                )
                is_success = False

        if not is_success:
            image_with_depth = self.default_generator()

        return image_with_depth, is_success


class VideoPathDataset(PathDataset):
    def __init__(
        self,
        clip_sampler,
        frame_sampler,
        decoder,
        normalize_to_0_1,
        *args,
        decoder_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_sampler = clip_sampler
        self.frame_sampler = frame_sampler
        self.decoder = decoder
        self.normalize_to_0_1 = normalize_to_0_1
        self.decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

    def _get_video_object(self, path):
        return EncodedVideo.from_path(
            path, decoder=self.decoder, decode_audio=False, **self.decoder_kwargs
        )

    def load_object(self, path) -> List[torch.Tensor]:
        """
        Returns:
            A (C, T, H, W) tensor.
        """
        video = self._get_video_object(path)
        # Read out all clips in this video
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = self.clip_sampler(
                end, video.duration, annotation=None
            )
            all_clips_timepoints.append((start, end))
        all_frames = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])["video"]
            if clip is None:
                logging.error(
                    "Got a None clip. Make sure the clip timepoints "
                    "are long enough: %s",
                    clip_timepoints,
                )
            frames = self.frame_sampler(clip)
            if self.normalize_to_0_1:
                frames = frames / 255.0  # since this is float, need 0-1
            all_frames.append(frames)
        if len(all_frames) == 1:
            # When only one clip is sampled (eg at training time), remove the
            # outermost list object so it can work with default collators etc.
            all_frames = all_frames[0]
        return all_frames

    def default_generator(self):
        dummy = (
            torch.ones(
                (
                    3,
                    self.frame_sampler._num_samples,
                    DEFAULT_SPATIAL_SIZE,
                    DEFAULT_SPATIAL_SIZE,
                )
            )
            * 0.5
        )
        if hasattr(self.clip_sampler, "_clips_per_video"):
            return [dummy] * self.clip_sampler._clips_per_video
        return dummy
