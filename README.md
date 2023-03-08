We provide a code example based on Autogluon.

## Installation
Installation follows the requirement for Autogluon.
```
./full_install.sh
```
## Example

We provide example on Petfinder. 

Data can be downloaded from:

https://drive.google.com/file/d/1AYDcrtfxlcIyvy9GzOripSR0QhItFTDG/view?usp=share_link

```
python petfinder.py
```
For turn on/off augmentation network and other configuration, please edit:

text/src/autogluon/text/automm/configs/model fusion_mlp_image_text_tabular.yaml
text/src/autogluon/text/automm/configs/optimization/adamw.yaml

## Citing LeMDA
BibTeX entry:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2212.14453,
  doi = {10.48550/ARXIV.2212.14453},
  url = {https://arxiv.org/abs/2212.14453},
  author = {Liu, Zichang and Tang, Zhiqiang and Shi, Xingjian and Zhang, Aston and Li, Mu and Shrivastava, Anshumali and Wilson, Andrew Gordon},
  title = {Learning Multimodal Data Augmentation in Feature Space},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## License

This library is licensed under the MIT License.
