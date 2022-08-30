# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/generic/util.py

import torch


def convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.
    """
    assert (
        torch.max(targets).item() < classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = torch.zeros(
        (targets.shape[0], classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


def maybe_convert_to_one_hot(
    target: torch.Tensor, model_output: torch.Tensor
) -> torch.Tensor:
    """
    This function infers whether target is integer or 0/1 encoded
    and converts it to 0/1 encoding if necessary.
    """
    target_shape_list = list(target.size())

    if len(target_shape_list) == 1 or (
        len(target_shape_list) == 2 and target_shape_list[1] == 1
    ):
        target = convert_to_one_hot(target.view(-1, 1), model_output.shape[1])

    # target are not necessarily hard 0/1 encoding. It can be soft
    # (i.e. fractional) in some cases, such as mixup label
    assert (
        target.shape == model_output.shape
    ), "Target must of the same shape as model_output."

    return target


def is_on_gpu(model: torch.nn.Module) -> bool:
    """
    Returns True if all parameters of a model live on the GPU.
    """
    assert isinstance(model, torch.nn.Module)
    on_gpu = True
    has_params = False
    for param in model.parameters():
        has_params = True
        if not param.data.is_cuda:
            on_gpu = False
    return has_params and on_gpu
