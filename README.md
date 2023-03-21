<p align="center">
  <img src="https://github.com/rederoth/ScanDy/blob/main/docs/scandy_repo_card.png">
</p>
<p align="center">
  <a href="https://github.com/psf/black">
  	<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://doi.org/10.1101/2023.03.14.532608">
    <img alt="paper" src="https://img.shields.io/badge/preprint-10.1101%2F2023.03.14.532608-blue"></a>    
</p>
<!-- # ScanDy
Simulating Realistic Human Scanpaths in Dynamic Real-World Scenes -->

## Introduction

`ScanDy` is a modular and mechanistic computational framework for simulating realistic **scan**paths in **dy**namic real-world scenes. It is specifically designed to quantitatively test hypotheses about eye-movement behavior in videos.
Specifically, it can be used to demonstrate the influence of object-representations on gaze behavior by comparing object-based and location-based models.

For a visual guide of how `ScanDy` works, have a look at the [interactive notebook](examples/interactive_guide.ipynb) (also on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/interactive_guide.ipynb)) and the <a href="#examples">example usecases</a>.

## Software architecture

The structure of `ScanDy` is inspired by the `neurolib` framework, which is also used for parameter optimization and exploration.
<p align="center">
  <img src="https://github.com/rederoth/ScanDy/blob/main/docs/software_architecture.png">
</p>
Scanpath models inherit from the `Model` base class, whose functionality includes initializing and running model simulations and the evaluation and visualization of the resulting scanpaths. Models are implemented in a modular way, consiting of moules for (I) Scene features, (II) Visual sensitivity, (III) Scanpath history, (IV) Decision making, and (V) Gaze update.

## Installation

You can install `ScanDy` as pypi package using `pip`:

```
pip install scandy
```

We however reccomend that you clone (or fork) this repository and install all dependencies with

```
git clone https://github.com/rederoth/ScanDy.git
cd neurolib/
pip install -r requirements.txt
pip install .
```

This gives you more freedom to modify the existing models and run the examples.

## Dataset

The scanpath models require precomputed maps of the video data. We use the VidCom dataset (Li et al., 2011), for which we provide all the required data on OSF (https://www.doi.org/10.17605/OSF.IO/83XUC).

To prepare the dataset, we used the following resources:

* [VidCom](http://ilab.usc.edu/vagba/dataset/VidCom/) - Video and eye-tracking data
* [deep_em_classifier](https://github.com/MikhailStartsev/deep_em_classifier/) - Eye movement classification
* [detectron2](https://github.com/facebookresearch/detectron2/) - Frame-wise object segmentation
* [deep_sort](https://github.com/nwojke/deep_sort/) - Object tracking
* [dynamic-proto-object-saliency](https://github.com/csmslab/dynamic-proto-object-saliency/) - Low-level saliency maps
* [TASED-Net](https://github.com/MichiganCOG/TASED-Net/) - High-level saliency maps
* [PWC-Net](https://github.com/NVlabs/PWC-Net/) - Optical flow calculation

If you only want to play around with a single video, we uploaded a version of the dataset only containing the "field03" video to [Google drive](https://drive.google.com/drive/folders/1ICTD9ENnidJXxHSvz30Aag3WXw8eslSZ?usp=share_link).

## Examples

We prepared a number of [IPython Notebooks](examples/) for you to explore the framework.

To get started with `ScanDy`, have a look at our [interactive guide](examples/interactive_guide.ipynb), where you can explore the effect of individual model parameters.

Additionally, we show instructive usecases, including:

* [Example 1](examples/ex1_scanpath_sgl_video.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex1_scanpath_sgl_video.ipynb): Scanpath simulation and visualization for a single video
* [Example 2](examples/ex2_model_comparison.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex2_model_comparison.ipynb): Evolutionary optimization of model parameters
* [Example 3](examples/ex3_model_extension.ipynb), on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/ex3_model_extension.ipynb): Extending on existing models: Location-based model with object-based sensitivity

All figures from our manuscript (Roth et al., 2023) can be reproduced with [this notebook](examples/manuscript_results.ipynb), which is also executable on [Colab](https://colab.research.google.com/github/rederoth/ScanDy/blob/main/examples/manuscript_results.ipynb).

## More information

### How to cite

If `ScanDy` is useful for your research, please cite our preprint:
> Roth, N., Rolfs, M., Hellwich, O., & Obermayer, K. (2023). Objects guide human gaze behavior in dynamic real-world scenes. *bioRxiv*, 2023-03.

```bibtex
@article{roth2023objects,
 title = {Objects Guide Human Gaze Behavior in Dynamic Real-World Scenes},
 author = {Roth, Nicolas and Rolfs, Martin and Hellwich, Olaf and Obermayer, Klaus},
 elocation-id = {2023.03.14.532608},
 year = {2023},
 doi = {10.1101/2023.03.14.532608},
 publisher = {Cold Spring Harbor Laboratory},}
```

### Contact

 If you have feedback, questions, and/or ideas, feel free to send a [mail](mailto:roth@tu-berlin.de) to Nico.

Nicolas Roth,
PhD Student at Science of Intelligence;
Neural Information Processing Group,
Fakultaet IV, Technische Universitaet Berlin,
MAR 5-6, Marchstr. 23, 10587 Berlin

### Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC 2002/1 "Science of Intelligence" – project number 390523135.
