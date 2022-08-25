# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class OmniOptimizer(object):
    def __init__(self, optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        for _, set_of_schedulers in enumerate(self.schedulers):
            for option, _ in set_of_schedulers.items():
                assert option in self.optimizer.defaults, (
                    "Optimizer option "
                    f"{option} not found in {self.optimizer}. Valid options are "
                    f"{self.optimizer.defaults.keys()}"
                )

    def step_schedulers(self, where: float) -> None:
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                new_value = scheduler(where)
                param_group[option] = new_value

    def step(self, where, closure=None):
        self.step_schedulers(where)
        return self.optimizer.step(closure)

    def zero_grad(self):
        return self.optimizer.zero_grad()
