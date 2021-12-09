# Neural Marionette

## Introduction
This is an official pytorch code for the paper, Neural Marionette: Unsupervised Learning of Motion Skeleton and Latent Dynamics for Volumetric Video (AAAI 2022).

This work is to extract skeletal structure from volumetric observations and to learn motion dynamics from the detected skeletal motions in a fully unsupervised manner.

Our model conducts <strong>motion generation/interpolation/retargeting</strong> based on the learned latent dynamics.

## Install
We tested on Python 3.8 and Ubuntu 18.04 LTS.

The architecture is built from Pytorch 1.7.1 with Cuda 11.0.

Creating a conda environment is recommended.

```shell
## Download the repository
git clone https://github.com/jinseokbae/neural_motion_learner.git
cd neural_motion_learner
## Create conda env
conda create --name nmotion python=3.8
conda activate nmotion
## modify setup.sh to match your cuda setting
bash setup.sh
```

## Run
Using provided pretrained model, run demo codes to visualize followings:
```shell
## Motion generation
python vis_generation.py
## Result will be stored in output/generation
```
![Gen Video](gifs/generation_demo.gif)

```shell
## Motion interpolation
python vis_interpolation.py
## Result will be stored in output/interpolation
```
![Interp Video](gifs/interpolation_demo.gif)

```shell
## Motion retargeting
python vis_retarget.py
## Result will be stored in output/retarget
```
![Retarget Video](gifs/retarget_demo.gif)

