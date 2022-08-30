## Omnivision Trainer

Training pipeline supporting Omnivore and OmniMAE projects.

## Installation
Omnivision requires Python 3.9. To install PyTorch 1.12 with CUDA 11.3 on Python 3.9 via conda, run the following instructions -

```bash
conda create --name ov python=3.9
conda activate ov
conda install pytorch=1.12 torchvision=0.13 cudatoolkit=11.3 -c pytorch
```

Install Omnivision in developer mode (where it runs code from your local checkout and picks up your changes) by running -
```bash
git clone https://github.com/facebookresearch/omnivore.git
cd omnivore
pip install -e ".[dev]"
```

## Testing
Before running the tests, please ensure that you installed the necessary additional test dependencies.

Use the the following command to run the tests:
```bash
# run all tests
python -m unittest discover -v -s tests -t .
# run a specific test
python -m unittest tests.test_scheduler
```

## Data preparation
All our dataloaders rely on `.npy` numpy array files for the meta data.

For IN1k, K400 and SSv2 datasets, please create two seperate `.npy` files for each split(train and val), such that there is one file consisting of a 1D arrays of image/video paths and another file consisting of the corresponding labels.. 

For SunRGBD, prepare three seperate `.npy` files for each split(train and val), such that there is one file consisting of a 1D array of image paths, one file consiting of a 1D array of corresponding depth image paths and the another file consisting of the corresponding labels.

Post that, update the `config/experiments/dataset_catalog.yaml` file with the paths to your newly created  `.npy` files for each dataset.

For instance, a sample numpy file for images or depth images or videos would look like this,
```
array(['/path_to_sample_1.JPEG', # .mp4, .avi, .png any such extensions are supported based on the data type.
       '/path_to_sample_2.JPEG',
       '/path_to_sample_3.JPEG',
       '/path_to_sample_4.JPEG',
       '/path_to_sample_5.JPEG',
       dtype='<U75')
```

And a sample numpy file for labels would look like this,
```
array([86, 2, 34, 48, 51]) # consisting of integer labels.
```

## Usage
All our given configs are designed to work on SLURM. We tested our configs with V100-32GB GPUS.
For locally running the configs and for quick debuging, append the following lines to your job commands.

```
submitit.use_cluster=false launcher.gpus_per_node=1 launcher.num_nodes=1
```

Adittionally, update the SLURM config in `config/experiments/base.yaml` to reflect your enviroments partitions, constraints, etc.

### Omnivore
For training a Swin model jointly on IN1k, K400 and SunRGBD, please follow the example below.
```
CONFIG_SAN=omnivore/swin_train_in1k_k400_sun_concat EXP_DIR=<YOUR EXPERIMENT LOG DIRECTORY> && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR
```

For evaluating the released model checkpoints, follow the SwinT In1k inference example,
```
CONFIG_SAN=omnivore/inference_in1k_pretrained EXP_DIR=<YOUR EXPERIMENT LOG DIRECTORY> && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR
```

### OmniMAE
For omnivorous MAE-style pretraining on SSV2 and IN1k, please follow the example below,
```
CONFIG_SAN=omnimae/omnimae_vitbase_ssv2_in1k EXP_DIR=<YOUR EXPERIMENT LOG DIRECTORY> && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR
```

For finetuning the OmniMAE model, please follow these examples,
- Finetuning on IN1k from the above generated pretraining checkpoints,
```
CONFIG_SAN=omnimae/omnimae_vitbase_ft_in1k EXP_DIR=<YOUR EXPERIMENT LOG DIRECTORY> && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR \
pretrained_omnimae_checkpoint_path=<PATH TO PRE-TRAINED OMNIMAE CHECKPOINT FROM ABOVE>
```

- Finetuning on SSV2 from the above generated pretraining checkpoints,
```
CONFIG_SAN=omnimae/omnimae_vitbase_ft_ssv2 EXP_DIR=<YOUR EXPERIMENT LOG DIRECTORY> && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR \
pretrained_omnimae_checkpoint_path=<PATH TO PRE-TRAINED OMNIMAE CHECKPOINT FROM ABOVE>
```

- Finetuning on IN1k from the OSS-released pretraining checkpoints,
```
CONFIG_SAN=omnimae/omnimae_vitbase_ft_from_oss_in1k EXP_DIR=/tmp/oss_rel_ft/ && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR \
pretrained_omnimae_checkpoint_path=<PATH TO PRE-TRAINED OMNIMAE CHECKPOINT FROM MODEL ZOO>h \
``

For evaluating the released model checkpoints, follow the VitB In1k inference example,
```
CONFIG_SAN=omnimae/inference_in1k_pretrained EXP_DIR=<YOUR EXPERIMENT LOG DIRECTORY> && \
python train_app_submitit.py +experiments=$CONFIG_SAN ++launcher.experiment_log_dir=$EXP_DIR
```

## Formatting
We use ufmt to handle formatting
```bash
ufmt --help
ufmt format
```

## TODOs

- [ ] Fix resumptions when using layer decay.
- [ ] Support EMA model in trainer.
