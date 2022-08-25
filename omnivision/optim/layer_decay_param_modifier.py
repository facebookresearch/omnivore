# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List


class ValueScaler(object):
    def __init__(self, scheduler, mult_val: float):
        self.scheduler = scheduler
        self.mult_val = mult_val

    def __call__(self, *args, **kwargs):
        val = self.scheduler(*args, **kwargs)
        return val * self.mult_val


def layer_decay_param_modifier(
    scheduler_cfgs: List[List[Dict]], model, layer_decay_value: float
) -> List[List[Dict]]:
    """
    Args
    - scheduler_cfgs: a list of omegaconf.ListConfigs.
        Each element in the list is a omegaconfg.DictConfig with the following structure
        {
            "scheduler": <some fvcore scheduler>
            "option": <value> possible options are "lr", "weight_decay" etc.
            "parameter_names": Set of str indicating param names that this scheduler applies to
        }
    - model: a model that implements a method `get_layer_id` that maps layer_name to an integer
    - layer_decay_value: float
    Returns
    - scheduler_configs: same structure as the input, elements can be modified
    """
    # FIXME: make sure the model API supports this
    num_layers = model.trunk.get_num_layers() + 1
    layer_decays = [
        layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)
    ]
    final_scheduler_cfgs = []
    # scheduler_cfgs is a list of lists
    for scheduler_cfg_group in scheduler_cfgs:
        curr_cfg_group = []
        # scheduler_cfg_group is a list of dictionaries
        for scheduler_cfg in scheduler_cfg_group:
            if scheduler_cfg["option"] != "lr":
                curr_cfg_group.append(scheduler_cfg)
                continue
            # Need sorted so that the list of parameter names is deterministic and consistent
            # across re-runs of this job. Else it was causing issues with loading the optimizer
            # state during a job restart (D38591759)
            parameter_names = sorted(scheduler_cfg["parameter_names"])
            for param_name in parameter_names:
                layer_id = model.trunk.get_layer_id(param_name)
                this_scale = layer_decays[layer_id]
                curr_param = {
                    "option": scheduler_cfg["option"],
                    "scheduler": ValueScaler(scheduler_cfg["scheduler"], this_scale),
                    "parameter_names": {param_name},
                }
                curr_cfg_group.append(curr_param)
        final_scheduler_cfgs.append(curr_cfg_group)
    return final_scheduler_cfgs
