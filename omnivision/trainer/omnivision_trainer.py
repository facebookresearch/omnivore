# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omnivision.data.api import Sample
from omnivision.data.concat_dataset import ConcatDataset
from omnivision.data.torch_dataset import TorchDataset
from omnivision.losses import wrap_base_loss
from omnivision.optim import construct_optimizer
from omnivision.utils.train import (
    AverageMeter,
    copy_data_to_device,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    is_dist_avail_and_initialized,
    makedir,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
    setup_logging,
)


def chunk_batch_for_accum_steps(batch, accum_steps):
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def get_chunk_from_data(data, chunk_id, num_chunks):
    """
    Recursively splits all the tensors inside the passed data object into num_chunks.
    """
    if isinstance(data, torch.Tensor):
        assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    elif isinstance(data, Sample):
        data_cls = type(data)
        data = data.__dict__
        return data_cls(**get_chunk_from_data(data, chunk_id, num_chunks))
    else:
        return data


@dataclass
class OmnivisionOptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OmnivisionOptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OmnivisionOptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OmnivisionOptimAMPConf(**self.amp)


@dataclass
class OmnivisionDistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False


@dataclass
class OmnivisionCudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False


@dataclass
class OmnivisionCheckpointConf:
    save_dir: str
    save_freq: int
    model_weight_initializer: Any = None


class OmnivisionTrainer(object):
    """
    Omnivision Trainer supporting the DDP training strategy.
    """

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        ## TODO: Re-factor to expose train_step as target.
        ## TODO: Support for Sync batchnorm.

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = logging
        self.checkpoint_conf = OmnivisionCheckpointConf(**checkpoint)
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.optim_conf = OmnivisionOptimConf(**optim or {})
        self.metrics_conf = metrics
        self.loss_conf = loss
        distributed = OmnivisionDistributedConf(**distributed or {})
        cuda = OmnivisionCudaConf(**cuda or {})

        self._maybe_infer_distributed_backend(distributed, accelerator)

        self._setup_env_variables(env_variables)
        self._setup_device(accelerator)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.local_rank,
        )
        # TODO: Enable seperate seed setting for each data worker.
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        self._setup_torch_dist_and_backend(cuda, distributed)

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizer()
        self.load_checkpoint()
        self._setup_ddp_components(distributed, accelerator)
        self._setup_dataloaders()
        dist.barrier()

    def _maybe_infer_distributed_backend(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        setup_distributed_backend(distributed_conf.backend)

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_components(self, distributed_conf, accelerator):
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )

        if distributed_conf.comms_dtype is not None:  # noqa

            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )
        self.model.to(self.device)

        if self.loss:
            copy_data_to_device(self.loss, self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)
        if self.metrics:
            self.metrics.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def checkpoint_save(self, epoch):

        if self.distributed_rank != 0:
            return

        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        checkpoint_paths = []
        checkpoint_paths.append(os.path.join(checkpoint_folder, "checkpoint.pt"))
        if (
            self.checkpoint_conf.save_freq > 0
            and int(epoch) % self.checkpoint_conf.save_freq == 0
        ):
            checkpoint_paths.append(
                os.path.join(checkpoint_folder, f"checkpoint_{int(epoch)}.pt")
            )

        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        for checkpoint_path in checkpoint_paths:
            with g_pathmgr.open(checkpoint_path, "wb") as f:
                torch.save(checkpoint, f)

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)

        if ckpt_path is None:
            # Loading pre-trained model weights
            model_weight_initializer = instantiate(
                self.checkpoint_conf.model_weight_initializer
            )
            if model_weight_initializer is not None:
                logging.info(
                    f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
                )
                self.model = model_weight_initializer(model=self.model)

        else:
            # Resuming from previous training checkpoint
            logging.info(f"Resuming training from {ckpt_path}")
            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")

            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
            self.loss.load_state_dict(checkpoint["loss"], strict=True)
            self.epoch = checkpoint["epoch"]
            self.steps = checkpoint["steps"]

            if self.optim_conf.amp.enabled and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):

        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get("val", None))
            if self.val_dataset:
                assert isinstance(
                    self.val_dataset, (TorchDataset, ConcatDataset)
                ), f"Unsuported Val dataloader: {type(self.val_dataset).__name__}"

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)
            assert isinstance(
                self.train_dataset, (TorchDataset, ConcatDataset)
            ), f"Unsuported Train dataloader: {type(self.train_dataset).__name__}"

    def run_train(self):
        # loop
        while self.epoch < self.max_epochs:

            outs = self.train_epoch(
                self.train_dataset.get_loader(epoch=int(self.epoch))
            )
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Run val
            if self.epoch % self.val_epoch_freq == 0:
                self.run_val()

            self.epoch += 1
            self.checkpoint_save(self.epoch)

    def run_val(self):

        if not self.val_dataset:
            return

        outs = self.val_epoch(self.val_dataset.get_loader(epoch=int(self.epoch)))
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader):

        batch_time = AverageMeter("Time", self.device, ":6.2f")
        data_time = AverageMeter("Data", self.device, ":6.2f")
        mem = AverageMeter("Mem (GB)", self.device, ":6.1f")
        phase_type = "val"

        iters_per_epoch = len(val_loader)

        metric_names = []
        if self.metrics_conf and phase_type in self.metrics_conf:
            for key in self.metrics_conf[phase_type].keys():
                for name in self.metrics_conf[phase_type][key]:
                    metric_names.append(f"Metrics/{phase_type}_{key}/{name}")

        metrics_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in metric_names]
        )

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *metrics_mts.values()],
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        if hasattr(self.model.module, "on_validation_epoch_start"):
            self.model.module.on_validation_epoch_start()

        end = time.time()

        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):

            if data_iter > limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            key, batch = self._process_batch(batch, phase_type)
            batch = copy_data_to_device(batch, self.device)

            # compute output
            with torch.no_grad():
                _, metrics_dict, batch_size = self._step(
                    batch,
                    key,
                    phase_type=phase_type,
                )

            for k in metrics_dict:
                metrics_mts[k].update(metrics_dict[k].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        progress.synchronize()
        self._reset_metrics("val")

        if hasattr(self.model.module, "on_validation_epoch_end"):
            self.model.module.on_validation_epoch_end()
        return {k: v.avg for k, v in metrics_mts.items()}

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Time", self.device, ":6.2f")
        data_time = AverageMeter("Data", self.device, ":6.2f")
        mem = AverageMeter("Mem (GB)", self.device, ":6.1f")
        phase_type = "train"

        iters_per_epoch = len(train_loader)

        metric_names = []
        if self.metrics_conf and phase_type in self.metrics_conf:
            for key in self.metrics_conf[phase_type].keys():
                for name in self.metrics_conf[phase_type][key]:
                    metric_names.append(f"Metrics/{phase_type}_{key}/{name}")

        metrics_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in metric_names]
        )

        loss_names = []
        for key in self.loss.keys():
            loss_names.append(f"Losses/{phase_type}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )

        # TODO: Track optimizer params (LR, WD,) etc.
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *metrics_mts.values(), *loss_mts.values()],
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()

        end = time.time()

        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )

        for data_iter, batch in enumerate(train_loader):

            if data_iter > limit_train_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            key, batch = self._process_batch(batch, phase_type)
            batch = copy_data_to_device(batch, self.device)

            accum_steps = batch.accum_steps
            chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self.optim.zero_grad()

            for i, chunked_batch in enumerate(chunked_batches):
                ddp_context = (
                    self.model.no_sync()
                    if i < accum_steps - 1
                    else contextlib.nullcontext()
                )

                with ddp_context:
                    with torch.cuda.amp.autocast(
                        enabled=self.optim_conf.amp.enabled,
                        dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                    ):
                        loss_dict, metrics_dict, batch_size = self._step(
                            chunked_batch,
                            key,
                            phase_type=phase_type,
                        )

                    assert len(loss_dict) == 1
                    loss_key, loss = loss_dict.popitem()

                    if not math.isfinite(loss.item()):
                        print("Loss is {}, stopping training".format(loss.item()))
                        sys.exit(1)

                    loss /= accum_steps
                    self.scaler.scale(loss).backward()

                    for k in metrics_dict:
                        metrics_mts[k].update(metrics_dict[k].item(), batch_size)

                    loss_mts[loss_key].update(loss.item(), batch_size)

            # compute gradient and do SGD step
            exact_epoch = self.epoch + float(data_iter) / iters_per_epoch

            self.optim.step_schedulers(float(exact_epoch) / self.max_epochs)

            if self.clip_gradient_partial is not None:
                self.scaler.unscale_(self.optim.optimizer)
                self.clip_gradient_partial(parameters=self.model.parameters())

            self.scaler.step(self.optim.optimizer)
            self.scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        progress.synchronize()
        self._reset_metrics("train")
        out_dict = {k: v.avg for k, v in metrics_mts.items()}
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        return out_dict

    def _compute_metrics(
        self, pred: torch.Tensor, label: torch.Tensor, phase_type: str, key: str
    ) -> Dict[str, torch.Tensor]:
        if self._get_metric_key(phase_type) not in self.metrics:
            return {}
        metrics_dict = self.metrics[self._get_metric_key(phase_type)][key]
        metrics_result = {}
        for name, metric in metrics_dict.items():
            metrics_result[f"Metrics/{phase_type}_{key}/{name}"] = metric(pred, label)
        return metrics_result

    def _reset_metrics(self, phase_type: str) -> None:
        if self._get_metric_key(phase_type) not in self.metrics:
            return
        metrics_dict = self.metrics[self._get_metric_key(phase_type)]
        for k_metric in metrics_dict.values():
            for metric in k_metric.values():
                metric.reset()

    def _get_metric_key(self, phase):
        return f"{phase}_"

    def _setup_components(self):
        logging.info("Setting up components: Model, loss, optim, metrics etc.")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

        self.logger = instantiate(self.logging_conf.tensorboard_writer)

        self.model = instantiate(self.model_conf, _convert_="all")

        self.loss = {
            key: wrap_base_loss(el)
            for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
        }
        self.loss = nn.ModuleDict(self.loss)

        self.metrics = nn.ModuleDict()
        if self.metrics_conf:
            metrics = instantiate(self.metrics_conf, _convert_="all")
            for phase, phase_metrics in metrics.items():
                self.metrics[self._get_metric_key(phase)] = nn.ModuleDict()
                for key, metrics in phase_metrics.items():
                    self.metrics[self._get_metric_key(phase)][key] = nn.ModuleDict(
                        metrics
                    )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # FIXME: grad clip shouldn't be an object
        self.clip_gradient_partial = instantiate(self.optim_conf.gradient_clip)

        logging.info("Finished setting up components: Model, loss, optim, metrics etc.")

        self.checkpoint_conf
        self.optim_conf
        self.data_conf

    def _construct_optimizer(self):
        self.optim = (
            None
            if self.optim_conf is None
            else construct_optimizer(
                self.model,
                self.optim_conf.optimizer,
                self.optim_conf.options,
                self.optim_conf.param_group_modifiers,
            )
        )

    def _process_batch(self, batch, phase_type):
        assert isinstance(batch, Mapping)
        assert all(isinstance(v, Sample) for v in batch.values())
        assert len(batch) == 1
        return batch.popitem()

    def _step(self, batch: Any, key: str, phase_type: str):

        y_hat = self.model({key: batch}, **batch.model_fwd_kwargs)
        assert isinstance(y_hat, Mapping)
        assert len(y_hat) == 1
        key, y_hat = y_hat.popitem()
        loss = None
        batch_size = batch.label.shape[0]
        loss_str = f"Losses/{phase_type}_{key}_loss"
        if phase_type == "train":
            loss, y_hat = self.loss[key](y_hat, batch)
            self.logger.log(
                os.path.join("Step", loss_str),
                loss,
                self.steps[phase_type],
            )

        metrics_result = self._compute_metrics(y_hat, batch.label, phase_type, key)
        self.logger.log_dict(
            {os.path.join("Step", k): v for k, v in metrics_result.items()},
            self.steps[phase_type],
        )

        self.steps[phase_type] += 1

        return {loss_str: loss}, metrics_result, batch_size
