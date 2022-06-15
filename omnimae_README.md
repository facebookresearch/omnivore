# Omnivore: A Single Model for Many Visual Modalities

[[paper](TODO)][[website](TODO)]

<p align="center">
  <img width='1000' src="./.github/omnimae_approach.png"/>  
</p>

   **OmniMAE: Single Model Masked Pretraining on Images and Videos.** Transfomer-based architectures have become competitive across a variety of visual domains, most notably images and videos. While prior work has studied these modalities in isolation, having a common architecture suggests that one can train a single unified model for multiple visual modalities. Prior attempts at unified modeling typically use architectures tailored for vision tasks, or obtain worse performance compared to single modality models. In this work, we show that masked autoencoding can be used to train a simple Vision Transformer on images and videos, without requiring any labeled data. This single model learns visual representations that are comparable to or better than single-modality representations on both image and video benchmarks, while using a much simpler architecture. In particular, our single pretrained model can be finetuned to achieve 84.7% on ImageNet and 73.4% on the challenging Something Something-v2 video benchmark. Furthermore, this model can be learned by dropping 90% of the image and 95% of the video patches, enabling extremely fast training. Code and models will be released.


## Model Zoo 

We share checkpoints for the models in the OmniMAE paper. 
**Coming Soon (ViT-B, ViT-L, ViT-H checkpoints)**


## Citation

If this work is helpful in your research, please consider starring :star: us and citing:  

```bibtex
@inproceedings{girdhar2022omnivore,
  title={{OmniMAE: Single Model Masked Pretraining on Images and Videos}},
  author={Girdhar, Rohit and El-Nouby Alaa and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={TODO},
  year={2022}
}
```

## Contributing
We welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more information.

## License
Omnivore is released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details. However the Swin Transformer implementation is additionally licensed under the Apache 2.0 license (see [NOTICE](NOTICE) for additional details).

