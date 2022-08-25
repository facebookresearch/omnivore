# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Iterable


class OmniDataset(ABC):
    @abstractmethod
    def get_loader(self, *args, **kwargs) -> Iterable:
        pass
