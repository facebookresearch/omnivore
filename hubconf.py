#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


dependencies = ["torch"]

from omnivore.models import (  # noqa: F401, E402
    omnivore_swinB,
    omnivore_swinB_epic,
    omnivore_swinB_imagenet21k,
    omnivore_swinL_imagenet21k,
    omnivore_swinL_kinetics600,
    omnivore_swinS,
    omnivore_swinT,
)
