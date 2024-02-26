# Localized Randomized Smoothing for Collective Robustness Certification

<p align="left">
<img src="https://www.cs.cit.tum.de/fileadmin/_processed_/f/4/csm_localized_randomized_smoothing_bce645f525.png", width="100%">

This is the official reference implementation of 

["Localized Randomized Smoothing"](https://openreview.net/pdf?id=-k7Lvk0GpBl)  
Jan Schuchardt*, Tom Wollschläger*, Aleksandar Bojchevski, and Stephan Günnemann, ICLR 2023.

## Requirements
To install the requirements, execute
```
conda env create -f image_environment.yaml
conda env create -f graph_environment.yaml
```
This will create two separate conda environments (`localized_smoothing_images` and `localized_smoothing_graphs`).

## Installation
You can install this package via `pip install -e .`

## Data
We use four different datasets: Pascal VOC (2012), Cityscapes, CiteSeer, and Cora_ML.

### Pascal VOC (2012)
First download the main Pascal VOC (2012) dataset via [torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.VOCSegmentation.html#torchvision.datasets.VOCSegmentation).  
Then, include the trainaug annotations, as explained in Section 2.2 of the [DeepLabV3 README](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/README.md).  
Finally, place `data/trainaug.txt` in `.../pascal_voc/VOCdevkit/VOC2012/ImageSets/Segmentation`.

### CityScapes
Download the `gtFine_trainvaltest` dataset from the official [CityScapes website](https://www.cityscapes-dataset.com/downloads/).

### CiteSeer and Cora_ML.
Citeseer and Cora_ML can be found in `.npz` format within the `data` folder.

## Usage
In order to reproduce all experiments, you will need need to execute the scripts in `seml/scripts` using the config files provided in `seml/configs`.  
We use the [SLURM Experiment Management Library](https://github.com/TUM-DAML/seml), but the scripts are just standard sacred experiments that can also be run without a MongoDB and SLURM installation.  

After computing all certificates, you can use the notebooks in `plots` to recreate the figures from the paper.  
In case you do not want to run all experiments yourself, you can just run the notebooks while keeping the flag `overwrite=False` (our results are then loaded from the respective `data` subfolders).

For more details on which config files and plotting notebooks to use for recreating which figure from the paper, please consult [REPRODUCE_IMAGES.MD](./REPRODUCE_IMAGES.md) and [REPRODUCE_GRAPHS.MD](./REPRODUCE_GRAPHS.md)

## Cite
Please cite our paper if you use this code in your own work:

```
@InProceedings{Schuchardt2023_Localized,
    author = {Schuchardt, Jan and Wollschl{\"a}ger, Tom and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
    title = {Localized Randomized Smoothing for Collective Robustness Certification},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023}
}
```
