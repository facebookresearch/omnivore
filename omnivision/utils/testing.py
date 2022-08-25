# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import tempfile
from contextlib import contextmanager
from typing import Callable

import torch
import torch.distributed as dist


@contextmanager
def with_temp_files(count: int):
    """
    Context manager to create temporary files and remove them
    after at the end of the context
    """
    if count == 1:
        fd, file_name = tempfile.mkstemp()
        yield file_name
        os.close(fd)
    else:
        temp_files = [tempfile.mkstemp() for _ in range(count)]
        yield [t[1] for t in temp_files]
        for t in temp_files:
            os.close(t[0])


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """

    def gpu_test_decorator(test_function: Callable):
        @functools.wraps(test_function)
        def wrapped_test(*args, **kwargs):
            if torch.cuda.device_count() >= gpu_count:
                return test_function(*args, **kwargs)

        return wrapped_test

    return gpu_test_decorator


def init_distributed_on_file(world_size: int, gpu_id: int, sync_file: str):
    """
    Init the process group need to do distributed training, by syncing
    the different workers on a file.
    """
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl",
        init_method="file://" + sync_file,
        world_size=world_size,
        rank=gpu_id,
    )
