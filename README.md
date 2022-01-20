# Omnivore: A Single Model for Many Visual Modalities

[[paper](https://arxiv.org/abs/2201.08377)][[website](https://facebookresearch.github.io/omnivore)]

<p align="center">
  <img width='1000' src="./.github/fig1.jpg"/>  
</p>

   **OMNIVORE is a single vision model for many different visual modalities.** It learns to construct representations that are aligned across visual modalities, without requiring training data that specifies correspondences between those modalities. Using OMNIVORE’s shared visual representation, we successfully identify nearest neighbors of left: an image (ImageNet-1K validation set) in vision datasets that contain right: depth maps (ImageNet-1K training set), single-view 3D images (ImageNet-1K training set), and videos (Kinetics-400 validation set).


This repo contains the code to run inference with a pretrained model on an image, video or RGBD image. 


## Usage

### Setup and Installation   

```
conda create --name omnivore python=3.8
conda activate omnivore
conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge -c pytorch -c defaults apex
conda install pytorchvideo
```

To run the notebook you may also need to install the follwing: 

```
conda install jupyter nb_conda ipykernel
python -m ipykernel install --user --name omnivore
```

### Run Inference 

Follow the `inference_tutorial.ipynb` tutorial [locally](https://github.com/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb) or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb) for step by step instructions on how to run inference with an image, video and RGBD image.

## Model Zoo 


| Name      | IN1k Top 1 | Kinetics400 Top 1     | SUN RGBD Top 1     | Model   |
| :---        |    :----   |          :--- | :--- |:--- |
| Omnivore Swin T      | 80.4       | 78.2   |62.5   | [weights](https://dl.fbaipublicfiles.com/omnivore/models/swinT_checkpoint.torch)   
| Omnivore Swin S   | 82.7       | 81.5      |64.2  | [weights](https://dl.fbaipublicfiles.com/omnivore/models/swinS_checkpoint.torch)  |
| Omnivore Swin B      | 83.8       | 82.8   |64.2   | [weights](https://dl.fbaipublicfiles.com/omnivore/models/swinB_checkpoint.torch)   |
| Omnivore Swin B (IN21k)   | 85.2       | 83.2      |65.5   | [weights](https://dl.fbaipublicfiles.com/omnivore/models/swinB_In21k_checkpoint.torch)   |
| Omnivore Swin L (IN21k)      | 85.6       | 83.4   |67.4   | [weights](https://dl.fbaipublicfiles.com/omnivore/models/swinL_In21k_checkpoint.torch) |

Numbers are based on Table 2. and Table 4. in the Omnivore Paper.

### Torch Hub 

Models can be loaded via torch hub e.g. 

```
model = torch.hub.load("facebookresearch/omnivore", model="omnivore_swinB")
```

The class mappings for the datasets can be downloaded as follows: 

```
wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json 
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json 
wget https://dl.fbaipublicfiles.com/omnivore/sunrgbd_classnames.json
```

## Citation

If this work is helpful in your research, please consider starring :star: us and citing:  

```bibtex
@article{girdhar2022omnivore,
  title={{Omnivore: A Single Model for Many Visual Modalities}},
  author={Girdhar, Rohit and Singh, Mannat, and Ravi, Nikhila and van der Maaten, Laurens and Joulin, Armand and Misra, Ishan},
  journal={arXiv preprint arXiv:2201.08377},
  year={2022}
}
```

## Contributing
We welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more information.

## License
Omnivore is released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details. However the Swin Transformer implementation is additionally licensed under the Apache 2.0 license (see [NOTICE](NOTICE) for additional details).

