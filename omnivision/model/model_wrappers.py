# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
from omnivision.data.api import VisionSample


class MIMOHeadWrapper(nn.Module):
    """Attaches multiple input multiple output heads to the trunk using forward hooks.

    Args:
        trunk: Any model to which you want to attach the heads to.
        heads: A list of dicts with the following keys:
            fork_module: The module which the head will be applied to. It can be an
                empty string, in which case the head is attached to the trunk's output.
            head: The head which is to be attached.
            input_key: The head will only run on inputs with this key. If set to
                `None` the head will be applied to all inputs.
            output_key: The head will produce this output key. If set to `None`, the
                output key will be the same as the input key.

            An example heads value can look like -
            ```
            [
                {
                    "fork_module": "layer_1.layer_a.layer_alpha",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_1",
                    "output_key": "out_1",
                },
                {
                    "fork_module": "",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_1",
                    "output_key": "out_2",
                },
                {
                    "fork_module": "",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_2",
                    "output_key": "out_3",
                },
                {
                    "fork_module": "",
                    "head": nn.Conv2d(in_feat, out_feat),
                    "input_key": None,
                    "output_key": None,
                },
            ]
            ```
        trunk_fields: A list of dicts with the following keys:
            input_key: The input key this rule applies to. If `None`, applies to all
                inputs.
            args: These specific keys will be fetched from the sample and passed as
                *args to the trunk for the specified `input_key`.
            kwargs: These specific keys will be fetched from the sample and passed as
                **kwargs to the trunk for the specified `input_key`.

            Example -
            ```
            [
                {
                    "input_key": "dataset_1",
                    "args": ["vision"]
                },
                {
                    "input_key": "dataset_2",
                    "args": ["vision"],
                    "kwargs": {"mask": "mask"}
                },
            ]
            ```

        Note that two heads cannot produce the same output key in the same forward pass.

    Returns:
        A dict with keys corresponding to the output keys which match with the input key.
    """

    @dataclass
    class HeadArgs:
        fork_module: str
        head: nn.Module
        input_key: Optional[str]
        output_key: Optional[str]

    @dataclass
    class TrunkFieldArgs:
        input_key: Optional[str]
        args: List[str] = field(default_factory=list)
        kwargs: Dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        trunk: nn.Module,
        heads: List[Dict],
        trunk_fields: List[Dict],
        handle_list_inputs=False,
    ) -> None:
        """WARNING: handle_list_inputs is a hack which needs to be refactored away."""
        super().__init__()

        self.trunk = trunk
        self.handle_list_inputs = handle_list_inputs

        # cast to HeadArgs for input validation
        heads = [self.HeadArgs(**head_dict) for head_dict in heads]
        # cast to TrunkFieldArgs for input validation
        trunk_fields = [
            self.TrunkFieldArgs(**trunk_fields_dict)
            for trunk_fields_dict in trunk_fields
        ]

        self.head_name_to_fork_module = {}
        self.heads = nn.ModuleList()
        self.head_input_keys = []
        self.head_output_keys = []
        self.head_fork_modules = []

        for head_args in heads:
            self.heads.append(head_args.head)
            self.head_input_keys.append(head_args.input_key)
            self.head_output_keys.append(head_args.output_key)
            self.head_fork_modules.append(head_args.fork_module)

        self.trunk_field_args = {}
        self.trunk_field_kwargs = {}
        for trunk_fields_elem in trunk_fields:
            input_key = trunk_fields_elem.input_key
            if input_key in self.trunk_field_args:
                raise KeyError(
                    f"Multiple trunk_fields specified for the same input_key: {input_key}"
                )
            self.trunk_field_args[input_key] = trunk_fields_elem.args
            self.trunk_field_kwargs[input_key] = trunk_fields_elem.kwargs

        # outputs is used as a temporary storage of the head outputs
        self.outputs = {}

        # input_key is used to specify which key is currently being processed
        self.input_key = None

        # handles to the hooks which can be used for removing the hooks if needed
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for i, head in enumerate(self.heads):
            fork_module_name = self.head_fork_modules[i]

            def hook_fn(
                module,
                module_in,
                module_out,
                # the following variables are passed as kwargs in the closure to avoid
                # late binding in python
                head_method=head,
                in_key=self.head_input_keys[i],
                out_key=self.head_output_keys[i],
            ):
                if in_key is not None and self.input_key != in_key:
                    return
                if out_key is None:
                    out_key = self.input_key
                if out_key in self.outputs:
                    # reset state before raising
                    self.outputs = {}
                    self.input_key = None
                    raise ValueError(
                        f"Two heads produced the same output key `{out_key}` during forward"
                    )
                self.outputs[out_key] = head_method(module_out)

            fork_module = self.trunk.get_submodule(fork_module_name)
            self.hook_handles.append(fork_module.register_forward_hook(hook_fn))

    def _get_trunk_fields(self):
        fields_args = self.trunk_field_args.get(self.input_key)
        fields_kwargs = self.trunk_field_kwargs.get(self.input_key)
        if fields_args is None:
            assert fields_kwargs is None
            fields_args = self.trunk_field_args.get(None)
            fields_kwargs = self.trunk_field_kwargs.get(None)
            if fields_args is None:
                assert fields_kwargs is None
                raise ValueError(
                    f"No trunk fields specified for input key: {self.input_key}"
                )
        return fields_args, fields_kwargs

    def forward_sub_batch(self, sub_batch, *args, **kwargs):
        assert isinstance(sub_batch, VisionSample), f"Received {type(sub_batch)}"
        fields_args, fields_kwargs = self._get_trunk_fields()
        sample_args = [getattr(sub_batch, arg) for arg in fields_args]
        sample_kwargs = {
            key: getattr(sub_batch, field) for key, field in fields_kwargs.items()
        }
        self.trunk(*sample_args, *args, **sample_kwargs, **kwargs)

    def forward(self, batch, *args, **kwargs) -> Dict:
        assert isinstance(batch, Mapping)
        assert len(self.outputs) == 0
        for key, sub_batch in batch.items():
            self.input_key = key
            if self.handle_list_inputs and isinstance(sub_batch.vision, Sequence):
                # FIXME: this only handles list inputs for the field "vision"
                assert len(batch) == 1
                out_vals = []
                for e in sub_batch.vision:
                    e_batch = copy.copy(sub_batch)
                    e_batch.vision = e
                    self.forward_sub_batch(e_batch, *args, **kwargs)
                    assert len(self.outputs) == 1
                    out_key, out_val = self.outputs.popitem()
                    out_vals.append(out_val)
                return {out_key: torch.cat(out_vals)}
            else:
                self.forward_sub_batch(sub_batch, *args, **kwargs)
        outputs = self.outputs
        self.input_key = None
        self.outputs = {}
        return outputs
