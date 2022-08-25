# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
from pathlib import Path

import torch
import torch.nn as nn
from omnivision.data.api import VisionSample
from omnivision.model.model_wrappers import MIMOHeadWrapper
from tests.util import SimpleNet


CONFIG_FOLDER = Path(__file__).parent / "configs_model_wrapper"


class TestMIMOHeadWrapper(unittest.TestCase):
    def test_single_input_single_output(self):
        batch_size = 8
        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)

        heads = [
            {
                "head": nn.Linear(in_features=4, out_features=10),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_1",
            }
        ]
        trunk_fields = [{"input_key": None, "args": ["vision"]}]
        model = MIMOHeadWrapper(model, heads, trunk_fields)

        inp = {"input_1": VisionSample(vision=torch.rand(batch_size, 2))}

        out = model(inp)
        self.assertSetEqual(set(out.keys()), {"out_1"})
        self.assertEqual(out["out_1"].shape[0], batch_size)
        self.assertEqual(out["out_1"].shape[1], 10)

    def test_single_input_multi_output(self):
        batch_size = 8
        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)

        heads = [
            {
                "head": nn.Linear(in_features=4, out_features=10),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_1",
            },
            {
                "head": nn.Linear(in_features=4, out_features=2),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_2",
            },
        ]
        trunk_fields = [{"input_key": None, "args": ["vision"]}]
        model = MIMOHeadWrapper(model, heads, trunk_fields)

        inp = {"input_1": VisionSample(vision=torch.rand(batch_size, 2))}

        out = model(inp)
        self.assertSetEqual(set(out.keys()), {"out_1", "out_2"})
        self.assertEqual(out["out_1"].shape[0], batch_size)
        self.assertEqual(out["out_1"].shape[1], 10)
        self.assertEqual(out["out_2"].shape[0], batch_size)
        self.assertEqual(out["out_2"].shape[1], 2)

    def test_multi_input_multi_output(self):
        batch_size = 8
        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)

        heads = [
            {
                "head": nn.Linear(in_features=4, out_features=10),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_1",
            },
            {
                "head": nn.Linear(in_features=4, out_features=2),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_2",
            },
            {
                "head": nn.Linear(in_features=5, out_features=3),
                "fork_module": "layer_2",
                "input_key": "input_2",
                "output_key": "out_3",
            },
        ]
        trunk_fields = [{"input_key": None, "args": ["vision"]}]
        model = MIMOHeadWrapper(model, heads, trunk_fields)

        inp_1 = {"input_1": VisionSample(vision=torch.rand(batch_size, 2))}
        inp_2 = {"input_2": VisionSample(vision=torch.rand(batch_size, 2))}

        # run the same tests twice to make sure there are no internal state issues
        for _ in range(2):
            inp = dict(**inp_1, **inp_2)
            out = model(inp)
            self.assertSetEqual(set(out.keys()), {"out_1", "out_2", "out_3"})
            self.assertEqual(out["out_1"].shape[0], batch_size)
            self.assertEqual(out["out_1"].shape[1], 10)
            self.assertEqual(out["out_2"].shape[0], batch_size)
            self.assertEqual(out["out_2"].shape[1], 2)
            self.assertEqual(out["out_3"].shape[0], batch_size)
            self.assertEqual(out["out_3"].shape[1], 3)

            out = model(inp_1)
            self.assertSetEqual(set(out.keys()), {"out_1", "out_2"})
            self.assertEqual(out["out_1"].shape[0], batch_size)
            self.assertEqual(out["out_1"].shape[1], 10)
            self.assertEqual(out["out_2"].shape[0], batch_size)
            self.assertEqual(out["out_2"].shape[1], 2)

            out = model(inp_2)
            self.assertSetEqual(set(out.keys()), {"out_3"})
            self.assertEqual(out["out_3"].shape[0], batch_size)
            self.assertEqual(out["out_3"].shape[1], 3)

    def test_multi_input_multi_output_same_output_key(self):
        batch_size = 8
        num_layers = 4
        model = SimpleNet(2, num_layers, 1.0)

        heads = [
            {
                "head": nn.Linear(in_features=4, out_features=10),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_1",
            },
            {
                "head": nn.Linear(in_features=4, out_features=2),
                "fork_module": "layer_1",
                "input_key": "input_1",
                "output_key": "out_2",
            },
            {
                "head": nn.Linear(in_features=5, out_features=3),
                "fork_module": "layer_2",
                "input_key": "input_2",
                "output_key": "out_2",
            },
        ]
        trunk_fields = [{"input_key": None, "args": ["vision"]}]
        model = MIMOHeadWrapper(model, heads, trunk_fields)

        inp_1 = {"input_1": VisionSample(vision=torch.rand(batch_size, 2))}
        inp_2 = {"input_2": VisionSample(vision=torch.rand(batch_size, 2))}

        # second and third head will produce the same output key (out_2)
        # this should raise
        inp = dict(**inp_1, **inp_2)
        with self.assertRaises(Exception):
            out = model(inp)

        out = model(inp_1)
        self.assertSetEqual(set(out.keys()), {"out_1", "out_2"})
        self.assertEqual(out["out_1"].shape[0], batch_size)
        self.assertEqual(out["out_1"].shape[1], 10)
        self.assertEqual(out["out_2"].shape[0], batch_size)
        self.assertEqual(out["out_2"].shape[1], 2)

        out = model(inp_2)
        self.assertSetEqual(set(out.keys()), {"out_2"})
        self.assertEqual(out["out_2"].shape[0], batch_size)
        self.assertEqual(out["out_2"].shape[1], 3)
