#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="omnivore",
    version="0.0",
    author="FAIR",
    url="https://github.com/facebookresearch/omnivore",
    install_requires=[
        "einops",
        "timm",
        "ftfy",
        "regex",
        "torchmetrics",
        "torchaudio>=0.9.0",
        "hydra-core",
        "submitit>=1.4.4",
        "pytorchvideo>=0.1.5",
        "fvcore",
        "opencv-python",
        "tensorboard==2.9.1",
        "torch>=1.12",
        "torchvision>=0.13",
    ],
    license="CC BY-NC 4.0",
    tests_require=[],
    extras_require={
        "dev": [
            "sphinx",
            ##################################
            # Formatter settings based on
            # `pyfmt -V`
            "black==22.3.0",
            "ufmt==2.0.0b2",
            "usort==1.0.2",
            "libcst==0.4.1",
            ##################################
        ],
    },
    packages=find_packages(exclude=("scripts", "tests")),
)
