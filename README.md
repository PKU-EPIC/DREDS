# Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects (ECCV 2022)

![teaser](images/teaser.png)

## Introduction
This is the official repository of [**Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects**](https://arxiv.org). This paper investigates the problem of specular and transparent object depth simulation and restoration.

We propose a system composed of a RGBD fusion network **SwinDRNet** for depth restoration, along with a **synthetic data generation pipeline, Domain Randomization-Enhanced Depth Simulation (DREDS)**. The DREDS approach leverages domain randomization and active stereo depth sensor simulation, to generate a large-scale (130k) **synthetic RGBD dataset, DREDS**, containing photorealistic RGB images and simulated depths with realistic sensor noise. 

We also curate a **real-world dataset, STD**, that captures 30 cluttered scenes composed of 50 objects with various materials from specular, transparent, to diffuse.

Training on our simulated data, SwinDRNet can directly generalize to real RGBD images and significantly boosts the performance of perception and manipulation tasks (e.g. **category-level pose estimation, object grasping**)

For more information, please visit our [**project page**](https://github.com/PKU-EPIC).

## Overview
This repository provides:
- Blender-python code and asset of [Domain randomization-enhanced depth sensor simulator](https://github.com/PKU-EPIC/DREDS/blob/main/DepthSensorSimulator)
- PyTorch code and weights of [Depth restoration network SwinDRNet](https://github.com/PKU-EPIC/DREDS/blob/main/SwinDRNet)
- PyTorch code and weights of SwinDRNet baseline for category-level pose estimation
<!-- - PyTorch code and weights of [SwinDRNet baseline for category-level pose estimation](https://github.com/PKU-EPIC/DREDS/blob/main/DownstreamTasks) -->
- [The simulated DREDS dataset, and real STD dataset](https://github.com/PKU-EPIC/DREDS#dataset)

## Dataset


## Citation
If you find our work useful in your research, please consider citing:

## Contact
If you have any questions, please open a github issue or contact us:

Qiyu Dai: qiyudai@pku.edu.cn, Jiyao Zhang: zhangjiyao@stu.xjtu.edu.cn