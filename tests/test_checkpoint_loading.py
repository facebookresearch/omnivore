# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import tempfile
import unittest
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from omnivision.model.checkpoint_utils import (
    load_checkpoint_and_apply_kernels,
    load_state_dict_into_model,
)
from tests.util import SimpleNet

CONFIG_FOLDER = Path(__file__).parent / "configs_checkpoint"


class TestCheckpointLoaderConf(unittest.TestCase):
    def test_simple_model(self):

        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)
        model_ckpt = SimpleNet(2, num_layers, 0.0)

        for i in range(4):
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_path = os.path.join(tmpdirname, "test.ckpt")
            ckpt = {"state_dict": model.state_dict()}
            torch.save(ckpt, ckpt_path)
            ckpt_st_dict = load_checkpoint_and_apply_kernels(checkpoint_path=ckpt_path)
            model_ckpt = load_state_dict_into_model(ckpt_st_dict, model_ckpt)

        for i in range(4):
            self.assertEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

    def test_include_filter_model(self):

        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)
        model_ckpt = SimpleNet(2, num_layers, 0.0)

        for i in range(4):
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        # add include, exclude filter
        filter_conf = OmegaConf.create(
            [
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptIncludeKernel",
                    "key_pattern": ["layer_[1,2]*"],
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_path = os.path.join(tmpdirname, "test.ckpt")
            ckpt = {"state_dict": model.state_dict()}
            torch.save(ckpt, ckpt_path)
            ckpt_st_dict = load_checkpoint_and_apply_kernels(
                checkpoint_path=ckpt_path,
                checkpoint_kernels=[hydra.utils.instantiate(f) for f in filter_conf],
            )
            model_ckpt = load_state_dict_into_model(
                ckpt_st_dict, model_ckpt, strict=False
            )

        for i in [1, 2]:
            self.assertEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        for i in [0, 3]:
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

    def test_exclude_filter_model(self):

        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)
        model_ckpt = SimpleNet(2, num_layers, 0.0)

        for i in range(4):
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        # add include, exclude filter
        filter_conf = OmegaConf.create(
            [
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptExcludeKernel",
                    "key_pattern": ["layer_[1,2]*"],
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_path = os.path.join(tmpdirname, "test.ckpt")
            ckpt = {"state_dict": model.state_dict()}
            torch.save(ckpt, ckpt_path)
            ckpt_st_dict = load_checkpoint_and_apply_kernels(
                checkpoint_path=ckpt_path,
                checkpoint_kernels=[hydra.utils.instantiate(f) for f in filter_conf],
            )
            model_ckpt = load_state_dict_into_model(
                ckpt_st_dict, model_ckpt, strict=False
            )

        for i in [0, 3]:
            self.assertEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        for i in [1, 2]:
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

    def test_include_exclude_filter_model(self):

        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)
        model_ckpt = SimpleNet(2, num_layers, 0.0)

        for i in range(4):
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        # add include, exclude filter
        filter_conf = OmegaConf.create(
            [
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptIncludeKernel",
                    "key_pattern": ["layer_[1,2]*"],
                },
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptExcludeKernel",
                    "key_pattern": ["layer_2*"],
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_path = os.path.join(tmpdirname, "test.ckpt")
            ckpt = {"state_dict": model.state_dict()}
            torch.save(ckpt, ckpt_path)
            ckpt_st_dict = load_checkpoint_and_apply_kernels(
                checkpoint_path=ckpt_path,
                checkpoint_kernels=[hydra.utils.instantiate(f) for f in filter_conf],
            )
            model_ckpt = load_state_dict_into_model(
                ckpt_st_dict, model_ckpt, strict=False
            )

        for i in [1]:
            self.assertEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        for i in [0, 2, 3]:
            self.assertNotEqual(
                getattr(model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

    def test_remap_with_repeat_filter_and_exclude_model(self):

        num_layers = 4
        ref_val = 1.0
        model = SimpleNet(2, num_layers, ref_val)

        class ComplexNetRepeat(nn.Module):
            def __init__(self):
                super(ComplexNetRepeat, self).__init__()

                self.complex_layer_0 = SimpleNet(
                    2, num_layers, init_val=2.0
                )  # output will be 6 dims
                self.complex_layer_1 = SimpleNet(
                    2, num_layers, init_val=3.0
                )  # output will be 10 dims

            def forward(self, x):
                x = self.complex_layer_0(x)
                x = self.complex_layer_1(x)
                return x

        model_ckpt = ComplexNetRepeat()

        # add include, exclude filter
        filter_conf = OmegaConf.create(
            [
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptRenameWithCopyKernel",
                    "key_pattern": ["*"],
                    "source_pattern": "",  # Note: Only First occurence in target is replaced
                    "target_patterns": ["complex_layer_0.", "complex_layer_1."],
                },
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptExcludeKernel",
                    "key_pattern": ["*layer_2*"],
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_path = os.path.join(tmpdirname, "test.ckpt")
            ckpt = {"state_dict": model.state_dict()}
            torch.save(ckpt, ckpt_path)
            ckpt_st_dict = load_checkpoint_and_apply_kernels(
                checkpoint_path=ckpt_path,
                checkpoint_kernels=[hydra.utils.instantiate(f) for f in filter_conf],
            )
            model_ckpt = load_state_dict_into_model(
                ckpt_st_dict, model_ckpt, strict=False
            )

        for i in [0, 1, 3]:
            self.assertEqual(
                getattr(model_ckpt.complex_layer_0, f"layer_{i}").weight.mean().item(),
                ref_val,
            )
            self.assertEqual(
                getattr(model_ckpt.complex_layer_0, f"layer_{i}").bias.mean().item(),
                ref_val,
            )
            self.assertEqual(
                getattr(model_ckpt.complex_layer_1, f"layer_{i}").weight.mean().item(),
                ref_val,
            )
            self.assertEqual(
                getattr(model_ckpt.complex_layer_1, f"layer_{i}").bias.mean().item(),
                ref_val,
            )

    def test_ckpt_with_remap_model(self):

        num_layers = 4

        class BaseModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNet(2, num_layers, 2.0)

            def forward(self, x):
                return self.model(x)

        model = BaseModule()
        model_ckpt = SimpleNet(2, num_layers, 1.0)

        for i in [0, 1, 2, 3]:
            self.assertNotEqual(
                getattr(model.model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertNotEqual(
                getattr(model.model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )

        # add include, exclude filter
        filter_conf = OmegaConf.create(
            [
                {
                    "_target_": "omnivision.model.checkpoint_utils.CkptRenameWithCopyKernel",
                    "key_pattern": ["model*"],
                    "source_pattern": "model.",  # Note: Only First occurence in target is replaced
                    "target_patterns": [""],
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            ckpt_path = os.path.join(tmpdirname, "test.ckpt")
            ckpt = {"state_dict": model.state_dict()}
            torch.save(ckpt, ckpt_path)
            ckpt_st_dict = load_checkpoint_and_apply_kernels(
                checkpoint_path=ckpt_path,
                checkpoint_kernels=[hydra.utils.instantiate(f) for f in filter_conf],
            )
            model_ckpt = load_state_dict_into_model(ckpt_st_dict, model_ckpt)

        for i in [0, 1, 2, 3]:
            self.assertEqual(
                getattr(model.model, f"layer_{i}").weight.mean(),
                getattr(model_ckpt, f"layer_{i}").weight.mean(),
            )
            self.assertEqual(
                getattr(model.model, f"layer_{i}").bias.mean(),
                getattr(model_ckpt, f"layer_{i}").bias.mean(),
            )
