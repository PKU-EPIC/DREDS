#!/bin/bash

# set working root and number of scene
cd /data/sensor/renderer/DepthSensorSimulator
scene_num_count=0;
end_scene_num=3000;

# run renderer.py
while (( $scene_num_count < $end_scene_num )); do
    /home/qiyudai/blender-2.93.3-linux-x64/blender material_lib_v2.blend --background --python renderer.py -- $scene_num_count;
((mycount=$scene_num_count+1));
done;