# Copyright Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from omnivision.optim import construct_optimizer, OmniOptimizer
from omnivision.trainer.omnivision_trainer import OmnivisionOptimConf

CONFIG_FOLDER = Path(__file__).parent / "configs"


class MiniNet(nn.Module):
    def __init__(self):
        super(MiniNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

        nn.init.constant_(self.fc1.weight, 1.0)
        nn.init.constant_(self.fc1.bias, 2.0)

        nn.init.constant_(self.fc2.weight, 3.0)
        nn.init.constant_(self.fc2.bias, 4.0)

        nn.init.constant_(self.fc3.weight, 5.0)
        nn.init.constant_(self.fc3.bias, 6.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1d = nn.Conv1d(5, 10, 2, groups=5)
        self.mn1 = MiniNet()
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.mn1(x)
        x = self.bn1(x)
        return x


class TestSchedulerConf(unittest.TestCase):
    def _check_valid(self, param_groups, expected_values):
        for param_group in param_groups:
            self.assertTrue(
                set(["lr", "weight_decay", "params"]).issubset(set(param_group.keys()))
            )
            lr = np.round(param_group["lr"], 2)
            wd = np.round(param_group["weight_decay"], 2)
            self.assertEqual(expected_values[lr][wd], set(param_group["params"]))

    def test_scheduler_base(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_base.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 4)

        expected_values = {
            0.2: {0.4: set([mini_net.fc1.weight]), 0.5: set([mini_net.fc1.bias])},
            0.3: {
                0.4: set([mini_net.fc2.weight, mini_net.fc3.weight]),
                0.5: set([mini_net.fc2.bias, mini_net.fc3.bias]),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_basic_param_module(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_basic_param_module.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 2)

        expected_values = {
            0.2: {0.4: set([mini_net.fc1.weight, mini_net.fc1.bias])},
            0.3: {
                0.4: set(
                    [
                        mini_net.fc2.weight,
                        mini_net.fc2.bias,
                        mini_net.fc3.weight,
                        mini_net.fc3.bias,
                    ]
                ),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_unspecified_defaults(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_unspecified_defaults.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 3)
        expected_values = {
            0.2: {0.0: set([mini_net.fc1.weight, mini_net.fc1.bias])},
            0.9: {
                0.4: set([mini_net.fc2.weight]),
                0.0: set([mini_net.fc3.weight, mini_net.fc2.bias, mini_net.fc3.bias]),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_basic_param_module(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_basic_param_module.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 2)

        expected_values = {
            0.2: {0.4: set([mini_net.fc1.weight, mini_net.fc1.bias])},
            0.3: {
                0.4: set(
                    [
                        mini_net.fc2.weight,
                        mini_net.fc2.bias,
                        mini_net.fc3.weight,
                        mini_net.fc3.bias,
                    ]
                ),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_non_constant(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_linear.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 4)

        expected_values = {
            0.2: {0.4: set([mini_net.fc1.weight]), 0.5: set([mini_net.fc1.bias])},
            0.3: {
                0.4: set([mini_net.fc2.weight, mini_net.fc3.weight]),
                0.5: set([mini_net.fc2.bias, mini_net.fc3.bias]),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)
        # check values at init are same for step where = 0
        optimizer.step(0.0)
        self._check_valid(optimizer.optimizer.param_groups, expected_values)

        optimizer.step(0.25)
        # Update the LR (key in expected_values) to correspond to 0.25 of training
        lr_0_25 = np.round((1 - 0.25) * 0.2, 2)
        expected_values[lr_0_25] = expected_values[0.2]
        del expected_values[0.2]
        self._check_valid(optimizer.optimizer.param_groups, expected_values)

        optimizer.step(0.5)
        # Update the LR (key in expected_values) to correspond to 0.5 of training
        lr_0_5 = np.round((1 - 0.5) * 0.2, 2)
        expected_values[lr_0_5] = expected_values[lr_0_25]
        del expected_values[lr_0_25]
        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_defaults(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_with_defaults.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 6)

        expected_values = {
            0.2: {0.4: set([mini_net.fc1.weight]), 0.5: set([mini_net.fc1.bias])},
            0.3: {0.4: set([mini_net.fc3.weight]), 0.5: set([mini_net.fc3.bias])},
            0.6: {0.4: set([mini_net.fc2.weight]), 0.5: set([mini_net.fc2.bias])},
        }
        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_multiple(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_with_multiple.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 4)
        expected_values = {
            0.2: {0.4: set([mini_net.fc1.weight]), 0.5: set([mini_net.fc1.bias])},
            0.3: {
                0.4: set([mini_net.fc2.weight, mini_net.fc3.weight]),
                0.5: set([mini_net.fc2.bias, mini_net.fc3.bias]),
            },
        }
        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_complex_param_module1(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "scheduler_complex_param_module_mixed1.yaml"
            )
        )
        mini_net = ComplexNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 3)

        expected_values = {
            0.2: {0.5: set([mini_net.mn1.fc1.weight, mini_net.mn1.fc1.bias])},
            0.3: {
                0.4: set(
                    [
                        mini_net.bn1.weight,
                        mini_net.bn1.bias,
                        mini_net.conv1d.weight,
                        mini_net.conv1d.bias,
                    ]
                ),
            },
            0.8: {
                0.5: set(
                    [
                        mini_net.mn1.fc2.weight,
                        mini_net.mn1.fc2.bias,
                        mini_net.mn1.fc3.weight,
                        mini_net.mn1.fc3.bias,
                    ]
                ),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_complex_param_module2(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "scheduler_complex_param_module_mixed2.yaml"
            )
        )
        mini_net = ComplexNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 3)

        expected_values = {
            0.2: {
                0.5: set(
                    [
                        mini_net.mn1.fc2.weight,
                        mini_net.mn1.fc2.bias,
                        mini_net.mn1.fc3.weight,
                        mini_net.mn1.fc3.bias,
                    ]
                )
            },
            0.3: {
                0.4: set(
                    [
                        mini_net.bn1.weight,
                        mini_net.bn1.bias,
                        mini_net.conv1d.weight,
                        mini_net.conv1d.bias,
                    ]
                ),
                0.5: set([mini_net.mn1.fc1.weight, mini_net.mn1.fc1.bias]),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_scheduler_complex_param_module3(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "scheduler_complex_param_module_mixed3.yaml"
            )
        )
        mini_net = ComplexNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 2)

        expected_values = {
            0.3: {
                0.4: set(
                    [
                        mini_net.bn1.weight,
                        mini_net.bn1.bias,
                        mini_net.conv1d.weight,
                        mini_net.conv1d.bias,
                    ]
                ),
                0.5: set(
                    [
                        mini_net.mn1.fc1.weight,
                        mini_net.mn1.fc1.bias,
                        mini_net.mn1.fc2.weight,
                        mini_net.mn1.fc2.bias,
                        mini_net.mn1.fc3.weight,
                        mini_net.mn1.fc3.bias,
                    ]
                ),
            },
        }

        self._check_valid(optimizer.optimizer.param_groups, expected_values)

    def test_invalid_scheduler_multiple_defaults(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "invalid_scheduler_multiple_defaults.yaml")
        )
        mini_net = MiniNet()
        with self.assertRaisesRegex(
            AssertionError, ".*one scheduler.*option.*default.*"
        ):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_invalid_scheduler_overlapping_groups(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "invalid_scheduler_overlapping_groups.yaml"
            )
        )
        mini_net = MiniNet()
        with self.assertRaisesRegex(AssertionError, ".*param.groups.*disjoint.*"):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_invalid_scheduler_overlapping_groups_module1(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "scheduler_complex_param_module_invalid1.yaml"
            )
        )
        mini_net = ComplexNet()
        with self.assertRaisesRegex(AssertionError, ".*param.groups.*disjoint.*"):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_invalid_scheduler_overlapping_groups_module2(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "scheduler_complex_param_module_invalid2.yaml"
            )
        )
        mini_net = ComplexNet()
        with self.assertRaisesRegex(AssertionError, ".*param.groups.*disjoint.*"):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_invalid_scheduler_nonexistent_module(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_param_module_non_existent.yaml")
        )
        mini_net = MiniNet()
        with self.assertRaisesRegex(
            AssertionError, ".*option.*lr.*BatchNorm1d.*not match.*class.*"
        ):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_unused_param_groups(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_unused_param_names.yaml")
        )
        mini_net = MiniNet()
        with self.assertRaisesRegex(
            AssertionError,
            ".*option.*lr.*fc4\*.*not match.*",
        ):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_unused_param_groups_multiple(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(
                CONFIG_FOLDER / "scheduler_unused_param_names_multiple.yaml"
            )
        )
        mini_net = MiniNet()
        with self.assertRaisesRegex(
            AssertionError,
            ".*option.*lr.*fc4\*.*not match.*",
        ):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_invalid_option(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_invalid_option.yaml")
        )
        mini_net = MiniNet()
        with self.assertRaisesRegex(AssertionError, ".*wd.*not found.*SGD.*"):
            construct_optimizer(
                mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
            )

    def test_scheduler_only_default(self) -> None:
        conf = OmnivisionOptimConf(
            **OmegaConf.load(CONFIG_FOLDER / "scheduler_only_default.yaml")
        )
        mini_net = MiniNet()
        optimizer = construct_optimizer(
            mini_net, conf.optimizer, conf.options, conf.param_group_modifiers
        )
        self.assertIsInstance(optimizer, OmniOptimizer)
        self.assertEqual(len(optimizer.optimizer.param_groups), 1)

        expected_values = {
            0.2: {
                0.4: set(
                    [
                        mini_net.fc1.weight,
                        mini_net.fc1.bias,
                        mini_net.fc2.weight,
                        mini_net.fc3.weight,
                        mini_net.fc2.bias,
                        mini_net.fc3.bias,
                    ]
                ),
            }
        }
        self._check_valid(optimizer.optimizer.param_groups, expected_values)
