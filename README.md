# ScanDy

Simulating Realistic Human Scanpaths in Dynamic Real-World Scenes

## Introduction

`ScanDy` is a modular and mechanistic computational framework for simulating realistic **scan**paths in **dy**namic real-world scenes. It is specifically designed to quantitatively test hypotheses about eye-movement behavior in videos.
Specifically, it can be used to demonstrate the influence of object-representations on gaze behavior by comparing object-based and location-based models.

For a visual guide of how `ScanDy` works, have a look at the [interactive notebook](https://github.com/rederoth/ScanDy/blob/main/examples/interactive_model_visu.ipynb) and the <a href="#example-usecases">example usecases</a>.

## Software architecture

The structure of `ScanDy` is inspired by the `neurolib` framework, which is also used for parameter optimization and exploration.
<p align="center">
  <img src="https://github.com/rederoth/ScanDy/software_architecture.png">
</p>
Scanpath models inherit from the `Model` base class, whose functionality includes initializing and running model simulations and the evaluation and visualization of the resulting scanpaths. Models are implemented in a modular way, consiting of moules for (I) Scene features, (II) Visual sensitivity, (III) Scanpath history, (IV) Decision making, and (V) Gaze update.

## Dataset

The scanpath models require precomputed maps of the video data. We use the VidCom dataset (Li et al., 2011), for which we provide the required data in the following folder: <https://tubcloud.tu-berlin.de/s/f7LTZATZXEjWwta>.

To prepare the dataset, we used the following resources:

* [VidCom](http://ilab.usc.edu/vagba/dataset/VidCom/) - Video and eye-tracking data
* [deep_em_classifier](https://github.com/MikhailStartsev/deep_em_classifier/) - Eye movement classification
* [detectron2](https://github.com/facebookresearch/detectron2/) - Frame-wise object segmentation
* [deep_sort](https://github.com/nwojke/deep_sort/) - Object tracking
* [dynamic-proto-object-saliency](https://github.com/csmslab/dynamic-proto-object-saliency/) - Low-level saliency maps
* [TASED-Net](https://github.com/MichiganCOG/TASED-Net/) - High-level saliency maps
* [PWC-Net](https://github.com/NVlabs/PWC-Net/) - Optical flow calculation

## Example usecases

To get started with `ScanDy`, have a look at the example [IPython Notebooks](examples/). There we show instructive usecases, including:

* [Example 1](examples/ex1_scanpath_sgl_video.ipynb): Scanpath simulation and visualization for a single video
* [Example 2](examples/ex2_model_comparison.ipynb): Evolutionary optimization of model parameters 
* [Example 3](examples/ex3_model_extension.ipynb): Extending on existing models: Location-based model with object-based sensitivity


## More information

### How to cite

If `ScanDy` is useful for your research, please cite our preprint:
> Roth, N., Rolfs, M., Hellwich, O., & Obermayer, K. (2023). Objects guide human gaze behavior in dynamic real-world scenes. *bioRxiv*, 2023-03.

```bibtex
@article{roth2023objects,
  title={Objects guide human gaze behavior in dynamic real-world scenes},
  author={Roth, Nicolas and Rolfs, Martin and Hellwich, Olaf and Obermayer, Klaus},
  journal={bioRxiv},
  pages={2023--03},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

### Contact

 Nicolas Roth [mail](mailto:roth@tu-berlin.de)

Cluster of Excellence Science of Intelligence & Institut für Softwaretechnik und Theoretische Informatik,
Technische Universität Berlin, Marchstraße 23, 10587 Berlin, Germany

### Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC 2002/1 "Science of Intelligence" – project number 390523135.
