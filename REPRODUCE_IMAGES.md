# Reproducing our experiments

In the following, we describe which scripts and notebooks to run in which order to reproduce the different image segmentation figures from our paper.

Note that you may have to adjust the directories in the individual config files (to point at the correct dataset folders, result folders etc.).  
You have to manually create these directories, they are not automatically created by the program itself.  
You will also need to adjust the slurm configuration parameters at the top of each file to match your cluster configuration (partition names etc.).

If you do not want to train and certify the models yourself, you can just run the plotting notebooks while keeping the flag `overwrite=False`.  
If you do, you will need to set `overwrite=True` when running the notebook for the first time.

## Pascal VOC

### Strictly local, mIOU vs avg. cert. radius (Fig. 2)
```
seml train_images_pascal_masked add seml/configs/segmentation/pascal/train_masked.yaml start
seml cert_images_pascal_masked add seml/configs/segmentation/pascal/cert_masked.yaml start
seml cert_images_pascal_masked add seml/configs/segmentation/pascal/cert_masked_center.yaml start
```
Then run `plots/pascal/masked/masked.ipynb`.

### Softly local, budget vs cert. accuracy (Fig. 3)
```
seml train_images_pascal add seml/configs/segmentation/pascal/train.yaml start
seml train_images_pascal_localized add seml/configs/segmentation/pascal/train_localized.yaml start
seml cert_images_pascal_iid add seml/configs/segmentation/pascal/cert_iid.yaml start
seml cert_images_pascal_localized_training add seml/configs/segmentation/pascal/cert_localized_training.yaml start
```
Then run `plots/pascal/normal/radius_vs_cert_acc.ipynb`.


### Softly local, mIOU vs avg. cert. radius (Fig. 4)
```
seml train_images_pascal add seml/configs/segmentation/pascal/train.yaml start
seml train_images_pascal_localized add seml/configs/segmentation/pascal/train_localized.yaml start
seml cert_images_pascal_iid add seml/configs/segmentation/pascal/cert_iid.yaml start
seml cert_images_pascal_localized_training add seml/configs/segmentation/pascal/cert_localized_training.yaml start
```
Then run `plots/pascal/normal/locally_trained.ipynb`.


## Cityscapes

### Softly Local, mIOU vs avg. cert. radius (Fig. 6)
```
seml train_images_cityscapes add seml/configs/segmentation/cityscapes/train_cityscapes.yaml start
seml cert_images_cityscapes_iid add seml/configs/segmentation/cityscapes/cert_iid_cityscapes.yaml start
seml cert_images_cityscapes add seml/configs/segmentation/pascal/cert_cityscapes.yaml start
```
Then run `plots/cityscapes/fewer_samples.ipynb`.