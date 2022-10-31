#!/usr/bin/env fbpython
# Copyright Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import subprocess
import unittest
from pathlib import Path

import torch


class TestOmnivoreJob(unittest.TestCase):
    def test_omnivore_job(self):
        # WARNING: This test is not run on sandcastle and only runs on OSS
        # or on a devserver

        if not torch.cuda.is_available():
            return

        omnivision_path = Path(__file__).resolve().parents[1] / "omnivision"

        launch_cmd = f"cd {omnivision_path} && "
        launch_cmd += "CONFIG_SAN=tests/swin_train_synthetic EXP_DIR=/tmp/omnivision_omnivore_tests/swin_train_synthetic.yaml/0 && "
        launch_cmd += "python train_app_submitit.py ++submitit.use_cluster=false "
        launch_cmd += "+experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR "
        launch_cmd += "launcher.gpus_per_node=1 launcher.num_nodes=1 trainer.data.train.num_workers=0 trainer.data.val.num_workers=0 "

        subprocess.run(launch_cmd, shell=True, check=True)
