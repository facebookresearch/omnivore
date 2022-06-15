# Omnivorous model architectures for Image and Video classfication and SSL 

This repository contains PyTorch evaluation code, pretrained models for the following papers:
<details>
<summary>
  <a href="omnivore_README.md">Omnivore</a> A single vision model for many different visual modalities, CVPR 2022 [<b>bib</b>]
</summary>

```
@inproceedings{girdhar2022omnivore,
  title={{Omnivore: A Single Model for Many Visual Modalities}},
  author={Girdhar, Rohit and Singh, Mannat and Ravi, Nikhila and van der Maaten, Laurens and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2022}
}
```
</details>
<details>
<summary>
<a href="omnimae_README.md">OmniMAE</a> Single Model Masked Pretraining on Images and Videos  [<b>bib</b>]
</summary>

```
@inproceedings{girdhar2022omnivore,
  title={{OmniMAE: Single Model Masked Pretraining on Images and Videos}},
  author={Girdhar, Rohit and El-Nouby Alaa and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={TODO},
  year={2022}
}
```
</details>


## Usage

### Setup and Installation   

Omnivore requires PyTorch and torchvision, please follow PyTorch's getting started [instructions](https://pytorch.org/get-started/locally/) for installation. If you are using conda on a linux machine, you can follow the following instructions -

```console
pip install .
```

This will install the required dependencies for you. You can alternatively install the required dependencies manually:

```console
conda create --name omnivore python=3.8
conda activate omnivore
conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=11.1 -c pytorch
```

We also require `einops`, `pytorchvideo` and `timm` which can be installed via pip -
```console
pip install einops
pip install pytorchvideo
pip install timm
```


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

# SSv2
wget TODO
```

The list of videos used for Kinetics-400 experiments can be found here: [training](http://dl.fbaipublicfiles.com/omnivore/kinetics400_lists/vidpaths_train.txt) and [validation](http://dl.fbaipublicfiles.com/omnivore/kinetics400_lists/vidpaths_val.txt).


### Run Inference 

Follow the `inference_tutorial.ipynb` tutorial [locally](https://github.com/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb) or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb) for step by step instructions on how to run inference with an image, video and RGBD image.

To run the tutorial you need to install `jupyter notebook`. For installing it on conda, you may run the follwing:

```
conda install jupyter nb_conda ipykernel ipywidgets
```

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/omnivore)

Replicate web demo and docker image is available! You can try loading with different checkpoints here
[![Replicate](https://replicate.com/facebookresearch/omnivore/badge)](https://replicate.com/facebookresearch/omnivore)


## Contributing
We welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more information.

## License
Omnivore is released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details. However the Swin Transformer implementation is additionally licensed under the Apache 2.0 license (see [NOTICE](NOTICE) for additional details).

