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


## Contributing
We welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more information.

## License
Omnivore is released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details. However the Swin Transformer implementation is additionally licensed under the Apache 2.0 license (see [NOTICE](NOTICE) for additional details).

