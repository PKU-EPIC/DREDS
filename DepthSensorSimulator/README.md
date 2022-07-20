# Domain Randomization-Enhanced Depth Sensor Simulator 

## Installation
- Download [Blender 2.93.3 (Linux x64)](https://download.blender.org/release/Blender2.93/blender-2.93.3-linux-x64.tar.xz) compressed file and uncompress it.
- Download the [environment map asset](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/simulator/envmap_lib.tar.gz) and the [blend file](https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/simulator/material_lib_v2.blend).

## File Structure
```
Depth Sensor Simulator
```

## Usage
- Run the shell script to start data generation.
```bash 
bash run.sh

```
- Conduct the stereo matching.
```bash 
python stereo_matching.py

```