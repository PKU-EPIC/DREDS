#!/bin/bash

# set working root and number of scene
cd /data/sensor/renderer/DepthSensorSimulator

# run renderer.py
# scene id: 0~2999
mycount=0;
while (( $mycount < 3000 )); do
    /home/qiyudai/blender-2.93.3-linux-x64/blender material_lib_v2.blend --background --python renderer.py -- $mycount;
((mycount=$mycount+1));
done;