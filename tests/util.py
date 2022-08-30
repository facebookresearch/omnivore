# Copyright (c) Meta Platforms, Inc. and affiliates.

import csv
import os
from functools import wraps
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

import torch
import torch.nn as nn
import torchvision.io as io


class SimpleNet(nn.Module):
    def __init__(self, inp_dim, num_layers, init_val=0.0):
        super().__init__()

        tmp_inp_dim = inp_dim
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, f"layer_{i}", nn.Linear(tmp_inp_dim, tmp_inp_dim + 1))
            layer = getattr(self, f"layer_{i}")
            nn.init.constant_(layer.weight, init_val)
            nn.init.constant_(layer.bias, init_val)
            tmp_inp_dim += 1

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
        return x


def create_small_kinetics_dataset(root_dir: str) -> None:
    """
    A test utility function to create a small Kinetics like dataset

    Args:
        root_dir(str): The directory to create the dataset in.
        Typically, a temporary directory is used.
    """
    video_codec = "libx264rgb"
    options = {"crf": "0"}
    height: int = 250
    width: int = 250
    num_frames = 20
    fps = 5
    data = create_dummy_video_frames(num_frames, height, width)

    train_data = [
        ["a.mp4", "308"],
        ["b.mp4", "298"],
        ["c.mp4", "240"],
        ["d.mp4", "363"],
    ]

    val_data = [
        ["a.mp4", "151"],
    ]

    for i in range(4):
        io.write_video(
            os.path.join(root_dir, train_data[i][0]),
            data,
            fps=fps,
            video_codec=video_codec,
            options=options,
        )

    train_file = os.path.join(root_dir, "train.csv")
    write_single_csv_file(train_file, train_data)

    val_file = os.path.join(root_dir, "val.csv")
    write_single_csv_file(val_file, val_data)


# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
def write_single_csv_file(file_name: str, data: List[Any]) -> None:
    with open(file_name, "w+", newline="") as csvfile:
        data_writer = csv.writer(
            # pyre-fixme[6]: Expected `_Writer` for 1st param but got `TextIOWrapper`.
            csvfile,
            delimiter=" ",
        )
        for row in data:
            data_writer.writerow(row)


# pyre-fixme[3]
def create_dummy_video_frames(num_frames: int, height: int, width: int):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())
    return torch.stack(data, 0)


def run_locally(func: Callable) -> Callable:  # pyre-ignore[24]
    """A decorator to run unittest locally."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # pyre-ignore[2,3]
        with mock.patch(
            "torch.distributed.is_available",
            return_value=False,
        ):
            return func(*args, **kwargs)

    return wrapper


def tempdir(func: Callable) -> Callable:  # pyre-ignore[24]
    """A decorator for creating a tempory directory that
    is cleaned up after function execution."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):  # pyre-ignore[2,3]
        with TemporaryDirectory() as temp:
            return func(self, temp, *args, **kwargs)

    return wrapper


def get_mock_init_trainer_params(
    overrides: Optional[Dict[str, Any]] = None,
) -> Callable[..., Dict[str, Any]]:
    """
    Order of trainer_params setting in unit test:
      - First call original function, which sets params from config
      - Then override some params to disable logger and checkpoint
      - Apply any test-specific overrides.
    """

    def mock_init_trainer_params(
        original: Callable[..., Dict[str, Any]],
    ) -> Dict[str, Any]:
        trainer_params = original()

        trainer_params["logger"] = False
        trainer_params["enable_checkpointing"] = False
        trainer_params["fast_dev_run"] = True

        if overrides:
            trainer_params.update(overrides)

        return trainer_params

    return mock_init_trainer_params
