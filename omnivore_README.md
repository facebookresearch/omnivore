# Omnivore: A Single Model for Many Visual Modalities

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnivore-a-single-model-for-many-visual/action-recognition-on-epic-kitchens-100)](https://paperswithcode.com/sota/action-recognition-on-epic-kitchens-100?p=omnivore-a-single-model-for-many-visual)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnivore-a-single-model-for-many-visual/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=omnivore-a-single-model-for-many-visual)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnivore-a-single-model-for-many-visual/scene-recognition-on-sun-rgbd)](https://paperswithcode.com/sota/scene-recognition-on-sun-rgbd?p=omnivore-a-single-model-for-many-visual)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnivore-a-single-model-for-many-visual/image-classification-on-inaturalist-2018)](https://paperswithcode.com/sota/image-classification-on-inaturalist-2018?p=omnivore-a-single-model-for-many-visual)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnivore-a-single-model-for-many-visual/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=omnivore-a-single-model-for-many-visual)

[[paper](https://arxiv.org/abs/2201.08377)][[website](https://facebookresearch.github.io/omnivore)]

<p align="center">
  <img width='1000' src="./.github/fig1.jpg"/>  
</p>

   **OMNIVORE is a single vision model for many different visual modalities.** It learns to construct representations that are aligned across visual modalities, without requiring training data that specifies correspondences between those modalities. Using OMNIVORE’s shared visual representation, we successfully identify nearest neighbors of left: an image (ImageNet-1K validation set) in vision datasets that contain right: depth maps (ImageNet-1K training set), single-view 3D images (ImageNet-1K training set), and videos (Kinetics-400 validation set).


This repo contains the code to run inference with a pretrained model on an image, video or RGBD image. 


## Model Zoo 

We share checkpoints for all the Omnivore models in the paper. The models are available via [torch.hub](https://pytorch.org/docs/stable/hub.html), and we also share URLs to all the checkpoints.

The details of the models, their torch.hub names / checkpoint links, and their performance is listed below.

| Name      | IN1k Top 1 | Kinetics400 Top 1     | SUN RGBD Top 1     | Model   |
| :---        |    :----   |          :--- | :--- |:--- |
| Omnivore Swin T      | 81.2       | 78.9   |62.3   | [omnivore_swinT](https://dl.fbaipublicfiles.com/omnivore/models/swinT_checkpoint.torch)   
| Omnivore Swin S   | 83.4       | 82.2      |64.6  | [omnivore_swinS](https://dl.fbaipublicfiles.com/omnivore/models/swinS_checkpoint.torch)  |
| Omnivore Swin B      | 84.0       | 83.3   |65.4   | [omnivore_swinB](https://dl.fbaipublicfiles.com/omnivore/models/swinB_checkpoint.torch)   |
| Omnivore Swin B (IN21k)   | 85.3       | 84.0      |67.2   | [omnivore_swinB_imagenet21k](https://dl.fbaipublicfiles.com/omnivore/models/swinB_In21k_checkpoint.torch)   |
| Omnivore Swin L (IN21k)      | 86.0       | 84.1   |67.1   | [omnivore_swinL_imagenet21k](https://dl.fbaipublicfiles.com/omnivore/models/swinL_In21k_checkpoint.torch) |

Numbers are based on Table 2. and Table 4. in the Omnivore Paper.

We also provide a torch.hub model/checkpoint file for an Omnivore model fine tuned on the Epic Kitchens 100 dataset: [omnivore_swinB_epic](https://dl.fbaipublicfiles.com/omnivore/models/swinB_epic_checkpoint.torch). 

##### Usage

The models can be loaded via torch.hub using the following command -

```
model = torch.hub.load("facebookresearch/omnivore", model="omnivore_swinB")
```

The class mappings for the datasets can be downloaded as follows: 

```
# Imagenet
wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json 

# Kinetics
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json 

# SUN RGBD
wget https://dl.fbaipublicfiles.com/omnivore/sunrgbd_classnames.json

# Epic Kitchens
wget https://dl.fbaipublicfiles.com/omnivore/epic_action_classes.csv
```

The list of videos used for Kinetics-400 experiments can be found here: [training](http://dl.fbaipublicfiles.com/omnivore/kinetics400_lists/vidpaths_train.txt) and [validation](http://dl.fbaipublicfiles.com/omnivore/kinetics400_lists/vidpaths_val.txt).



## Citation

If this work is helpful in your research, please consider starring :star: us and citing:  

```bibtex
@inproceedings{girdhar2022omnivore,
  title={{Omnivore: A Single Model for Many Visual Modalities}},
  author={Girdhar, Rohit and Singh, Mannat and Ravi, Nikhila and van der Maaten, Laurens and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2022}
}
```
