# SwinDRNet for Category-level Pose Estimation
PyTorch code and weights of SwinDRNet baseline for category-level pose estimation.
## System Dependencies
```bash
$ sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
$ sudo apt install libopenexr-dev zlib1g-dev openexr
```
## Setup
- ### Install pip dependencies
We have tested on Ubuntu 20.04 with an NVIDIA GeForce RTX 2080 and NVIDIA GeForce RTX 3090 with Python 3.7. The code may work on other systems.Install the dependencies using pip:
```bash
$ pip install -r requirments.txt
```
- ### Download dataset and models

1. Download the pre-trained model, and dataset. In the scripts below, be sure to comment out files you do not want, as they are very large. Alternatively, you can download files [manually](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/)

```bash
# Download DREDS and STD Dataset
$ cd data
$ bash DOWNLOAD.sh
$ cd ..

# Download the pretrained model
$ cd pretrained_model
$ bash DOWNLOAD.sh
$ cd ..

```
2. Model: We provide our pretrained model on here (will release soon). Please download to /results/ckpt/ .
3. Extract the downloaded dataset and merge the train split of DREDS-CatKnown following the file structure.
```
data
├── DREDS                              
│   ├── DREDS-CatKnown
│   │   ├── train
│   │   │   ├── 00001
│   │   │   └── ...
│   │   ├── val
│   │   │   ├── 01162
│   │   │   └── ...
│   │   └── test
│   │       ├── 00000
│   │       └── ...
│   └── DREDS-CatNovel
│       ├── 00029
│       └── ...
├── STD
│   ├── STD-CatKnown
│   │   ├── test_0
│   │   └── ...
│   └── STD-CatNovel
│       ├── test_novel_0-1
│       └── ...
└── cad_model
     ├──syn_train
     │    ├──00000000
     │    └──...
     ├──syn_test
     │    ├──00000000
     │    └──...
     ├──real_cat_known
     │    ├──aeroplane
     │    └──...
     └──real_cat_novel
          ├──0_trans_teapot
          └──...
```


## Training
- Start training by: 
    ```bash
    # An example command for training
    $ python train.py --train_data_path PATH_DRED_CatKnown_TrainSplit --val_data_path PATH_DRED_CatKnown_ValSplit --val_obj_path PATH_DRED_CatKnown_CADMOEL
    ```

## Testing 
- Start testing by: 
    ```bash
    # An example command for testing
    $ python inference.py --val_data_type TYPE_OF_DATA --train_data_path PATH_DRED_CatKnown_TrainSplit --val_data_path PATH_DRED_CatKnown_TestSplit  --val_obj_path PATH_DRED_CatKnown_CADMOEL --val_depth_path PATH_VAL_DEPTH
    ```