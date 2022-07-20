import cv2
import numpy as np
import os 
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
from utils.metrics_nocs import align, prepare_data_posefitting, draw_detections
from datasets.datasets import exr_loader,load_meta
syn_depth_path = '/data/sensor/data/real_data/test_0/0000_gt_depth.exr'
nocs_path = '/data/sensor/data/real_data/test_0/0000_coord.png'
mask_path = '/data/sensor/data/real_data/test_0/0000_mask.png'
meta_path = '/data/sensor/data/real_data/test_0/0000_meta.txt'
obj_dir = '/data/sensor/data/cad_model/real_cat_known'
synset_names = ['other',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'car',  # 5
                    'mug',  # 6
                    'aeroplane',  # 7
                    'BG',  # 8
                    ]
intrinsics = np.zeros((3,3))

img_h = 720 
img_w = 1280  
fx = 918.295227050781 
fy = 917.5439453125 

cx = img_w * 0.5 - 0.5
cy = img_h * 0.5 - 0.5
camera_params = {
    'fx': fx,
    'fy': fy,
    'cx': cx,
    'cy': cy,
    'yres': img_h,
    'xres': img_w,
}
hw = (640 / img_w, 360 / img_h)
camera_params['fx'] *= hw[0]
camera_params['fy'] *= hw[1]
camera_params['cx'] *= hw[0]
camera_params['cy'] *= hw[1]
camera_params['xres'] *= hw[0]
camera_params['yres'] *= hw[1]
intrinsics[0,0] = camera_params['fx']
intrinsics[0,2] = camera_params['cx']
intrinsics[1,1] = camera_params['fy']
intrinsics[1,2] = camera_params['cy']
intrinsics[2,2] = 1.0   


_syn_depth =  cv2.imread(syn_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
if len(_syn_depth.shape) == 3:
    _syn_depth = _syn_depth[:, :, 0]
coords = Image.open(nocs_path).convert('RGB')
coords = np.array(coords) / 255.

if mask_path.split('.')[-1] == 'exr':
    _mask = exr_loader(mask_path, ndim=1)
else:
    _mask = Image.open(mask_path)
    _mask = np.array(_mask)
if mask_path.split('.')[-1] == 'exr':
    _mask = np.array(_mask * 255, dtype=np.int32)   
_meta = load_meta(meta_path)

_mask_sem = np.full(_mask.shape, 8) #, 0）
_scale = np.ones((10,3)) #(class_num+1,3)
for i in range(len(_meta)):
    _mask_sem[_mask == _meta[i]["index"]] = _meta[i]["label"] #1
    if _meta[i]["instance_folder"] !=" " :

        bbox_file = os.path.join(obj_dir,_meta[i]["instance_folder"] ,_meta[i]["name"],'bbox.txt')
        bbox = np.loadtxt(bbox_file)
        _scale[_meta[i]["label"]] = bbox[0, :] - bbox[1, :]
    else :
        _scale[_meta[i]["label"]] = np.ones((3))
    _scale[_meta[i]["label"]] /= np.linalg.norm(_scale[_meta[i]["label"]])


gt_class_ids , gt_scores , gt_masks ,gt_coords ,\
                    gt_boxes = prepare_data_posefitting(_mask_sem,coords)

result = {}
result['gt_RTs'], scales, error_message, _ = align(gt_class_ids, 
                                                        gt_masks, 
                                                        gt_coords, 
                                                        _syn_depth, 
                                                        intrinsics, 
                                                        synset_names)

if 1:
    output_path = 'tmp'
    draw_rgb = False
    save_dir =os.path.join(output_path ,'save_{}'.format(i))
    if not os.path.exists(save_dir) :
        os.mkdir(save_dir)
    data = 'camera'
    result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
    rgb_path = '/data/sensor/data/real_data/test_0/0000_color.png'
    _rgb = Image.open(rgb_path).convert('RGB')
    _rgb = _rgb.resize((640,360))
    _rgb = np.array(_rgb)
    
    draw_detections(_rgb, save_dir, data, 1, intrinsics, synset_names, draw_rgb,
                            gt_boxes, gt_class_ids, gt_masks, gt_coords, result['gt_RTs'], scales, np.ones(gt_boxes.shape[0]),
                            gt_boxes, gt_class_ids, gt_masks, gt_coords, result['gt_RTs'], np.ones(gt_boxes.shape[0]), scales)