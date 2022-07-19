SwinDRNet
----------

This is the official implementation of SwinDRNet, a depth restoration network proposed in _["Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects"](https://eccv2022.ecva.net/)_. SwinDRNet takes inputs of a colored RGB image along with its aligned depth image and outputs a refined depth that restores the error area of the depth image and completes the invalid area caused by specular and transparent objects. The refined depth can be directly used for some downstream tasks (e.g., category-level object 6D pose estimation and robotic grasping). For more details, please see our paper and video.

![SwinDRNet](./images/SwinDRNet.png)

### System Dependencies
```bash
$ sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
$ sudo apt install libopenexr-dev zlib1g-dev openexr
```
### Setup
1. Install pip dependencies
```bash
$ pip install -r requirments.txt
```
2. Prepare dataset
### Training
```bash
# An example command for training
$ python train.py --train_data_path PATH_DRED_CatKnown_TrainSplit --val_data_path PATH_DRED_CatKnown_ValSplit
```
### Testing
```bash
# An example command for testing
$ python inference.py --train_data_path PATH_DRED_CatKnown_TrainSplit --val_data_path PATH_DRED_CatKnown_TestSplit
```
