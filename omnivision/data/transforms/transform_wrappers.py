# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
All Transform wrappers take as input a Sample, and return a Sample.
"""
import copy
from typing import Any, Callable, List

from omnivision.data.api import (
    dataclass_as_dict,
    Sample,
    VisionMaskSample,
    VisionSample,
)


class SingleFieldTransform(Callable):
    """
    The most basic transform, where only a single field is transformed. It wraps around
    any standard torchvision (or other) transformation function
    """

    def __init__(self, field: str, base_transform: object) -> None:
        super().__init__()
        self.field = field
        self.base_transform = base_transform

    def __call__(self, sample: Sample) -> Sample:
        setattr(sample, self.field, self.base_transform(getattr(sample, self.field)))
        return sample


class VisionTransform(SingleFieldTransform):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("vision", *args, **kwargs)


class ListTransform(Callable):
    """
    Apply transforms to a list of items
    """

    def __init__(self, base_transform: object, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_transform = base_transform

    def __call__(self, items: List[Any]) -> List[Any]:
        return [self.base_transform(item) for item in items]


class FlattenListOfList(Callable):
    """
    Flatten a list of list into a single longer list.
    """

    @staticmethod
    def __call__(all_samples: List[List[Any]]) -> List[Any]:
        return [sample for samples in all_samples for sample in samples]


class SingleFieldListToSampleList(Callable):
    """
    Convert a Sample with a list in the data to a list of Samples.
    """

    def __init__(self, field: str):
        self.field = field

    def __call__(self, sample: Sample) -> List[Sample]:
        data = getattr(sample, self.field)
        assert isinstance(data, list)
        delattr(sample, self.field)
        ret = []
        for el in data:
            new_sample = copy.deepcopy(sample)
            setattr(new_sample, self.field, el)
            ret.append(new_sample)
        return ret


class MaskingTransform(Callable):
    """
    Creates a mask for the input data. Useful for training MAE for instance.
    """

    def __init__(self, masking_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masking_object = masking_object

    def __call__(self, sample: VisionSample) -> VisionMaskSample:
        mask = self.masking_object(sample.vision)["mask"]
        return VisionMaskSample(mask=mask, **dataclass_as_dict(sample))
