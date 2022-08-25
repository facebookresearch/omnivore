#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copied from https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/generic/util.py

import logging
from typing import Dict, List, Optional

import torch
from iopath.common.file_io import g_pathmgr

from .distributed import broadcast_object, is_primary


# constants:
CHECKPOINT_FILE = "checkpoint.torch"
CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = torch.device("cuda")


def load_and_broadcast_checkpoint_list(
    checkpoint_paths: List[str], device: torch.device = CPU_DEVICE
):
    if is_primary():
        for path in checkpoint_paths:
            checkpoint = load_checkpoint(path, device)
            if checkpoint is not None:
                break
    else:
        checkpoint = None
    logging.info(f"Broadcasting checkpoint loaded from {checkpoint_paths}")
    return broadcast_object(checkpoint)


def load_and_broadcast_checkpoint(
    checkpoint_path: str, device: torch.device = CPU_DEVICE
) -> Optional[Dict]:
    """Loads a checkpoint on primary and broadcasts it to all replicas.

    This is a collective operation which needs to be run in sync on all replicas.

    See :func:`load_checkpoint` for the arguments.
    """
    if is_primary():
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        checkpoint = None
    logging.info(f"Broadcasting checkpoint loaded from {checkpoint_path}")
    return broadcast_object(checkpoint)


def load_checkpoint(
    checkpoint_path: str, device: torch.device = CPU_DEVICE
) -> Optional[Dict]:
    """Loads a checkpoint from the specified checkpoint path.

    Args:
        checkpoint_path: The path to load the checkpoint from. Can be a file or a
            directory. If it is a directory, the checkpoint is loaded from
            :py:data:`CHECKPOINT_FILE` inside the directory.
        device: device to load the checkpoint to

    Returns:
        The checkpoint, if it exists, or None.
    """
    if not checkpoint_path:
        return None

    assert device is not None, "Please specify what device to load checkpoint on"
    assert device.type in ["cpu", "cuda"], f"Unknown device: {device}"
    if device.type == "cuda":
        assert torch.cuda.is_available()

    if not g_pathmgr.exists(checkpoint_path):
        logging.warning(f"Checkpoint path {checkpoint_path} not found")
        return None
    if g_pathmgr.isdir(checkpoint_path):
        checkpoint_path = f"{checkpoint_path.rstrip('/')}/{CHECKPOINT_FILE}"

    if not g_pathmgr.exists(checkpoint_path):
        logging.warning(f"Checkpoint file {checkpoint_path} not found.")
        return None

    logging.info(f"Attempting to load checkpoint from {checkpoint_path}")
    # load model on specified device and not on saved device for model and return
    # the checkpoint
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=device)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint
