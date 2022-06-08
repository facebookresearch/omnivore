#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="omnivore_fair",
    version="1.0",
    author="FAIR",
    url="https://github.com/facebookresearch/omnivore",
    install_requires=[
        "einops",
        "pytorchvideo",
        "timm",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torchaudio>=0.9.0",
    ],
    license="CC BY-NC 4.0",
    tests_require=[],
    packages=find_packages(),
)
