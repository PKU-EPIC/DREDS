#!/bin/bash

# This script makes it easy to download DREDS/STD dataset.
# Files are at https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/

# Comment out any data you do not want.

echo 'Warning:  Files are *very* large.  Be sure to comment out any files you do not want.'


#----- DREDS Dataset -----------------------------------
mkdir DREDS
cd DREDS
mkdir DREDS-CatKnown
cd DREDS-CatKnown

wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/test/test.gz    # DREDS-CatKnown-Test (73.4G)
wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/val/val.tar.gz  # DREDS-CatKnown-Val (10.5G)

wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/train/train_part0.tar.gz  # DREDS-CatKnown-Train-Part0 (74.2G)
wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/train/train_part1.tar.gz  # DREDS-CatKnown-Train-Part1 (73.5G)
wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/train/train_part2.tar.gz  # DREDS-CatKnown-Train-Part2 (73.7G)
wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/train/train_part3.tar.gz  # DREDS-CatKnown-Train-Part3 (73.4G)
wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatKnown/train/train_part4.tar.gz  # DREDS-CatKnown-Train-Part4 (73.4G)

cd ..

wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/DREDS-CatNovel/DREDS-CatNovel.tar.gz  # DREDS-CatNovel (45.4G)

cd ..


#----- STD Dataset -----------------------------------
mkdir STD
cd STD

wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/STD-CatKnown/STD-CatKnown.tar.gz  # STD-CatKnown ()
wget https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/data/STD-CatNovel/STD-CatNovel.tar.gz  # STD-CatNovel ()

cd ..