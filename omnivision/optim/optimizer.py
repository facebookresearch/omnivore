# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import fnmatch
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, MISSING

from . import LARS, OmniOptimizer


def create_lars_optimizer(params, opt, **lars_params):
    optim = hydra.utils.instantiate(opt, params=params)
    return LARS(optim, **lars_params)


def validate_param_group_params(param_groups, model):
    parameters = [set(param_group["params"]) for param_group in param_groups]
    model_parameters = {parameter for _, parameter in model.named_parameters()}
    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Scheduler generated param_groups should be disjoint"
    assert (
        set.union(*parameters) == model_parameters
    ), "Scheduler generated param_groups include all parameters of the model"


def unix_pattern_to_parameter_names(
    scheduler_cfg: DictConfig, model: nn.Module
) -> Union[None, Set[str]]:
    if "param_names" not in scheduler_cfg and "module_cls_names" not in scheduler_cfg:
        return None
    return unix_param_pattern_to_parameter_names(scheduler_cfg, model).union(
        unix_module_cls_pattern_to_parameter_names(scheduler_cfg, model)
    )


def get_full_parameter_name(module_name, param_name):
    if module_name == "":
        return param_name
    return f"{module_name}.{param_name}"


def unix_module_cls_pattern_to_parameter_names(
    scheduler_cfg: DictConfig,
    model: nn.Module,
) -> Union[None, Set[str]]:
    if "module_cls_names" not in scheduler_cfg:
        return set()
    module_cls_to_params = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        module_cls_to_params.setdefault(module_cls, set())
        module_cls_to_params[module_cls] |= set(
            get_full_parameter_name(module_name, param_name)
            for param_name, _ in module.named_parameters()
        )
    parameter_names = []
    for module_cls_name in scheduler_cfg.module_cls_names:
        module_cls = hydra.utils.get_class(module_cls_name)
        matching_parameters = module_cls_to_params.get(module_cls, set())
        assert len(matching_parameters) > 0, (
            f"Optimizer option for {scheduler_cfg.option} module_cls_name"
            f" {module_cls_name} does not match any classes in the model"
        )
        logging.info(
            f"Matches for module_cls_name [{module_cls_name}]: {matching_parameters} "
        )
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def unix_param_pattern_to_parameter_names(
    scheduler_cfg: DictConfig,
    model: nn.Module,
) -> Union[None, Set[str]]:
    if "param_names" not in scheduler_cfg:
        return set()
    all_parameter_names = {name for name, _ in model.named_parameters()}
    parameter_names = []
    for param_name in scheduler_cfg.param_names:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert len(matching_parameters) >= 1, (
            f"Optimizer option for {scheduler_cfg.option} param_names {param_name} "
            "does not match any parameters in the model"
        )
        logging.info(f"Matches for param_name [{param_name}]: {matching_parameters}")
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def set_default_parameters(
    scheduler_cfgs: List[DictConfig], all_parameter_names: Set[str]
) -> None:
    constraints = [
        scheduler_cfg.parameter_names
        for scheduler_cfg in scheduler_cfgs
        if scheduler_cfg.parameter_names is not None
    ]
    if len(constraints) == 0:
        default_params = set(all_parameter_names)
    else:

        default_params = all_parameter_names - set.union(*constraints)
    default_count = 0
    for scheduler_cfg in scheduler_cfgs:
        if scheduler_cfg.parameter_names is None:
            scheduler_cfg.parameter_names = default_params
            default_count += 1
    assert default_count <= 1, "Only one scheduler per option can be default"
    if default_count == 0:  # Add defaults without options
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(
    param_constraints: List[Set[str]], model: torch.nn.Module
) -> List[torch.nn.Parameter]:
    matching_names = set.intersection(*param_constraints)
    return [value for name, value in model.named_parameters() if name in matching_names]


def map_scheduler_cfgs_to_param_groups(
    scheduler_cfgs_per_param_group: Iterable[List[Dict]], model: torch.nn.Module
) -> Tuple[List[Dict[Any, Any]], List[Dict[str, List[torch.nn.Parameter]]]]:
    schedulers = []
    param_groups = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        param_constraints = [
            scheduler_cfg["parameter_names"] for scheduler_cfg in scheduler_cfgs
        ]
        matching_parameters = name_constraints_to_parameters(param_constraints, model)
        if len(matching_parameters) == 0:  # If no overlap of parameters, skip
            continue
        schedulers_for_group = {
            scheduler_cfg["option"]: scheduler_cfg["scheduler"]
            for scheduler_cfg in scheduler_cfgs
            if "option" in scheduler_cfg
        }
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups


def construct_optimizer(
    model: torch.nn.Module,
    optimizer_conf,
    options_conf=None,
    param_group_modifiers_conf=None,
) -> OmniOptimizer:  # noqa
    """
    Constructs a stochastic gradient descent or ADAM (or ADAMw) optimizer
    with momentum. i.e, constructs a torch.optim.Optimizer with zero-weight decay
    Batchnorm and/or no-update 1-D parameters support, based on the config.

    Supports wrapping the optimizer with Layer-wise Adaptive Rate Scaling
    (LARS): https://arxiv.org/abs/1708.03888

    Args:
        model (nn.Module): model to perform stochastic gradient descent
            optimization or ADAM optimization.
        cfg (OptimizerConf): Hydra/Omega conf object consisting hyper-parameters
            of SGD or ADAM, includes base learning rate,  momentum, weight_decay,
            dampening and etc. The supported config schema is `OptimizerConf`.
    """
    if not options_conf:
        optimizer = hydra.utils.instantiate(optimizer_conf, params=model.parameters())
        return OmniOptimizer(optimizer)

    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_parameter_names = {name for name, _ in model.named_parameters()}
    flattened_scheduler_cfgs = []
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for config in scheduler_cfgs:
            config.option = option
            config.parameter_names = unix_pattern_to_parameter_names(config, model)
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        flattened_scheduler_cfgs.append(scheduler_cfgs)

    if param_group_modifiers_conf:
        for custom_param_modifier in param_group_modifiers_conf:
            custom_param_modifier = hydra.utils.instantiate(custom_param_modifier)
            flattened_scheduler_cfgs = custom_param_modifier(
                scheduler_cfgs=flattened_scheduler_cfgs, model=model
            )
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        itertools.product(*flattened_scheduler_cfgs), model
    )
    validate_param_group_params(param_groups, model)
    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return OmniOptimizer(optimizer, schedulers)
