# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code borrowed from TnT - https://github.com/pytorch/tnt/blob/master/torchtnt/loggers/tensorboard.py
import atexit
import logging
import os
import uuid
from typing import Any, Dict, Optional, Union

from numpy import ndarray
from omnivision.utils.train import get_machine_local_and_dist_rank, makedir
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):

    makedir(log_dir)
    return TensorBoardLogger(path=log_dir, **writer_kwargs)


# TODO: Expose writer building in configs.
class TensorBoardLogger(object):
    """
    A simple logger for TensorBoard.
    """

    def __init__(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """

        self._writer: Optional[SummaryWriter] = None

        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path

        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = SummaryWriter(
                log_dir=path, *args, filename_suffix=str(uuid.uuid4()), **kwargs
            )
        else:
            logging.debug(
                f"Not logging metrics on this host because env RANK: {self._rank} != 0"
            )

        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """

        if not self._writer:
            return

        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """

        if not self._writer:
            return

        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], metrics: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            metrics (dict): dictionary of name of metric and corersponding values
        """

        if not self._writer:
            return

        self._writer.add_hparams(hparams, metrics)

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None
