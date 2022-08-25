# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import torch


class LARS(torch.optim.Optimizer):
    """
    This class is adapted from
    https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py to
    include ignoring LARS application specific parameters (e.g. 1D params)

    Args:
        optimizer (torch.optim): Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr.
            See https://arxiv.org/abs/1708.03888
        clip (bool): Decides between clipping or scaling mode of LARS. If `clip=True` the
            learning rate is set to `min(optimizer_lr, local_lr)` for each parameter.
            If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps (float): epsilon kludge to help with numerical stability while calculating
        adaptive_lr.
        ignore_1d_param (float): If true, does not update 1 dimentional parameters.
    """

    def __init__(
        self,
        optimizer,
        trust_coefficient=0.02,
        clip=True,
        eps=1e-8,
        ignore_1d_param=True,
    ) -> None:
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip
        self.ignore_1d_param = ignore_1d_param

        self.defaults = self.optim.defaults

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self, closure=None):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group["weight_decay"] if "weight_decay" in group else 0
                weight_decays.append(weight_decay)
                apply_LARS = group["apply_LARS"] if "apply_LARS" in group else True
                if not apply_LARS:
                    continue
                group["weight_decay"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.ignore_1d_param and p.ndim == 1:  # ignore bias
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = (
                            self.trust_coefficient
                            * (param_norm)
                            / (grad_norm + param_norm * weight_decay + self.eps)
                        )

                        # clip learning rate for LARS
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied
                            # by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]
