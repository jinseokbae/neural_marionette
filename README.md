# Neural Motion Learner

## Introduction
This work is to extract skeletal structure from volumetric observations and to learn motion dynamics from the skeletal motions in fully unsupervised manner.

Our model conducts <strong>motion generation/interpolation/retargeting</strong> based on the learned latent dynamics.

Note that it is an <strong>unofficial version</strong> of the work so that minimal amounts of codes are provided to demonstrate results.

Full descriptions including title, training codes and data pre-processing methods will be uploaded once the paper of this work is accepted to the conference.

## Install
We tested on Python 3.8 with Ubuntu 18.04 LTS and Cuda 11.0.

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


```shell
## Motion interpolation
python vis_interpolation.py
## Result will be stored in output/interpolation
```

```shell
## Motion retargeting
python vis_retarget.py
## Result will be stored in output/retarget
```


