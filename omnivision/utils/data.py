# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os.path
import pickle
from multiprocessing import shared_memory
from typing import Any, List, Tuple, Union

import numpy as np
from iopath.common.file_io import g_pathmgr
from omnivision.utils.distributed import (
    barrier,
    broadcast_object,
    is_local_primary,
    is_torch_dataloader_worker,
)
from PIL import Image


class IdentityTransform:
    def __call__(self, x: Any) -> Any:
        return x


# copied from vissl.data.data_helper
def get_mean_image(crop_size: Union[Tuple, int]):
    """
    Helper function that returns a gray PIL image of the size specified by user.
    Args:
        crop_size (tuple, or int): used to generate (crop_size[0] x crop_size[1] x 3) image
            in the case of a tuple of (crop_size, crop_size, 3) image in case of int.
    Returns:
        img: PIL Image
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    img = Image.fromarray(
        128 * np.ones((crop_size[0], crop_size[1], 3), dtype=np.uint8)
    )
    return img


def list_of_paths_to_path(path_list: List[str]):
    path_exists = False
    for idx, path in enumerate(path_list):
        if g_pathmgr.exists(path):
            path_exists = True
            break
    if path_exists is False:
        path = None
    return path_exists, path, idx


def pickle_load(path):
    with g_pathmgr.open(path, "rb") as fh:
        data = pickle.load(fh)
    return data


def numpy_load(path):
    with g_pathmgr.open(path, "rb") as fh:
        data = np.load(fh)
    return data


FILE_EXT_TO_HANDLER = {".pkl": pickle_load, ".npy": numpy_load}


class FileLoader:
    @staticmethod
    def load(path_list: List[str], file_handler=None, return_idx=True):
        path_exists, path, idx = list_of_paths_to_path(path_list)
        if not path_exists:
            raise ValueError(f"No path exists in {path_list}")
        if file_handler is None:
            _, ext = os.path.splitext(path)
            file_handler = FILE_EXT_TO_HANDLER[ext]
        arr = file_handler(path)
        if return_idx:
            return arr, idx
        return arr


class SharedMemoryNumpyLoader:
    """
    WARN: A referenced to this object needs to be preserved till
    the returned np array is being used. This uses collective
    operations.
    """

    def __init__(self):
        self.sm = None
        self.sm_name = None

    def load(self, path_list: List[str]) -> np.ndarray:
        """Attempts to load data from a list of paths. Each element is tried (in order)
        until a file that exists is found. That file is then used to read the data.
        """
        if self.sm is not None:
            raise RuntimeError("Cannot load multiple objects with the same loader")

        path_exists, path, idx = list_of_paths_to_path(path_list)

        if not path_exists:
            raise ValueError(f"No path exists in {path_list}")

        self.sm_name = (
            "".join([x if x.isalnum() else "_" for x in path]) + f"_{getpass.getuser()}"
        )

        # we only read from local rank 0 parent process on a machine
        # all other GPU parent processes and dataloaders read from shared memory
        if is_local_primary() and not is_torch_dataloader_worker():
            # this is the local rank 0 process
            arr = load_file(path)
            assert isinstance(
                arr, np.ndarray
            ), f"arr is not an ndarray. found {type(arr)}"
            logging.info(f"Moving data files to shared memory: {self.sm_name}")
            try:
                sm = shared_memory.SharedMemory(
                    name=self.sm_name, create=True, size=arr.nbytes
                )
            except FileExistsError:
                logging.info(
                    "Shared memory already exists, closing it out and recreating"
                )
                sm_old = shared_memory.SharedMemory(name=self.sm_name, create=False)
                sm_old.close()
                sm_old.unlink()
                sm = shared_memory.SharedMemory(
                    name=self.sm_name, create=True, size=arr.nbytes
                )
            sm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=sm.buf)
            sm_arr[:] = arr[:]
            # barrier for all (non-dataloader) proceses to ensure the data is
            # available on all GPUs
            barrier()
            broadcast_object(sm_arr.shape)  # arr_shape
            broadcast_object(sm_arr.dtype)  # arr_type
        else:
            if not is_torch_dataloader_worker():
                # parent process on a GPU which isn't local rank 0; wait for barrier
                barrier()
                arr_shape = broadcast_object(None)
                arr_dtype = broadcast_object(None)
            logging.info(f"Loading data files from shared memory: {self.sm_name}")
            sm = shared_memory.SharedMemory(name=self.sm_name, create=False)
            sm_arr = np.ndarray(shape=arr_shape, dtype=arr_dtype, buffer=sm.buf)
        # need to keep a reference to the shared memory otherwise it will get
        # garbage collected and result in a segfault
        self.sm = sm
        return sm_arr, idx

    def __del__(self):
        # FIXME: this doesn't seem to be working on the FAIR cluster
        if self.sm is None:
            return
        self.sm.close()
        if is_local_primary() and not is_torch_dataloader_worker():
            logging.info(f"Unlinking shared memory: {self.sm_name}")
            self.sm.unlink()


# Copied from vissl.utils.io
def load_file(filename, mmap_mode=None):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    logging.info(f"Loading data from file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(fopen, encoding="latin1", mmap_mode=mmap_mode)
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without PathManager"
                )
                data = np.load(filename, encoding="latin1", mmap_mode=mmap_mode)
                logging.info("Successfully loaded without PathManager")
            except Exception:
                logging.info("Could not mmap without PathManager. Trying without mmap")
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(fopen, encoding="latin1")
        else:
            with g_pathmgr.open(filename, "rb") as fopen:
                data = np.load(fopen, encoding="latin1")
    elif file_ext == ".json":
        with g_pathmgr.open(filename, "r") as fopen:
            data = json.loads(fopen.read())
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def load_file_from_list(file_list, mmap_mode=None):
    for path in file_list:
        if g_pathmgr.exists(path):
            return load_file(path, mmap_mode)
            break
    raise Exception(f"None of the paths exist in {file_list}")
