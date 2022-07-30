'''Functions for reading and saving EXR images using OpenEXR.
'''

import sys
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn
from utils.api_utils import depth2rgb, label2color
import imageio
import os

sys.path.append('../..')

def seg_mask_to_rgb(seg_mask, num_classes):
    l2c = label2color(num_classes + 1)
    seg_mask_color = np.zeros((seg_mask.shape[0], 3, seg_mask.shape[2], seg_mask.shape[3]))
    for i in range(seg_mask.shape[0]):
        color = l2c.single_img_color(seg_mask[i])#.squeeze(2).transpose(2,0,1).unsqueeze(0)
        color = np.squeeze(color,axis=2)
        color = color.transpose((2,0,1))
        color = color[np.newaxis,:,:,:]
        seg_mask_color[i] = color
    seg_mask_color = torch.from_numpy(seg_mask_color)
    return seg_mask_color

def xyz_to_rgb(xyz_map):
    xyz_rgb = torch.ones_like(xyz_map)
    for i in range(xyz_rgb.shape[0]):
        xyz_rgb[i] = torch.div((xyz_map[i] - xyz_map[i].min()),
                               (xyz_map[i].max() - xyz_map[i].min()).item())
    return xyz_rgb    

def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    return camera_normal_rgb

def create_grid_image(inputs, outputs, labels, rgb, normal_pred, normal_labels, confidence_1, confidence_2, sem_masks=None, pred_sem_seg=None, coords=None, pred_coords=None, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    rgb_tensor = rgb[:max_num_images_to_save]
    normal_pred_tensor = normal_pred[:max_num_images_to_save]
    normal_pred_tensor = normal_to_rgb(normal_pred_tensor)
    normal_labels_tensor = normal_labels[:max_num_images_to_save]
    normal_labels_tensor = normal_to_rgb(normal_labels_tensor)

    if not coords is None:
        coords_tensor = coords[:max_num_images_to_save]
        pred_coords_tensor = pred_coords[:max_num_images_to_save]
        
        pred_sem_seg_tensor = pred_sem_seg[:max_num_images_to_save]
        sem_masks_tensor = sem_masks[:max_num_images_to_save]
        pred_sem_seg_tensor = seg_mask_to_rgb(pred_sem_seg_tensor, 8)
        sem_masks_tensor = seg_mask_to_rgb(sem_masks_tensor, 8)

    img_tensor = inputs[:max_num_images_to_save]
    output_tensor = outputs[:max_num_images_to_save]
    label_tensor = labels[:max_num_images_to_save]
    confidence_1_tensor = confidence_1[:max_num_images_to_save]
    confidence_2_tensor = confidence_2[:max_num_images_to_save]


    output_tensor[output_tensor < 0] = 0
    output_tensor[output_tensor > 4] = 0

    label_tensor[label_tensor < 0] = 0
    label_tensor[label_tensor > 4] = 4

    img_tensor = xyz_to_rgb(img_tensor)


    output_tensor = output_tensor.repeat(1, 3, 1, 1)
    label_tensor = label_tensor.repeat(1, 3, 1, 1)
    confidence_1_tensor = confidence_1_tensor.repeat(1, 3, 1, 1)
    confidence_2_tensor = confidence_2_tensor.repeat(1, 3, 1, 1)

    if coords is None:
        images = torch.cat((img_tensor, confidence_1_tensor, confidence_2_tensor, output_tensor, \
            label_tensor, rgb_tensor, normal_pred_tensor, normal_labels_tensor), dim=3)
    else:
        images = torch.cat((img_tensor, confidence_1_tensor, confidence_2_tensor, output_tensor, \
            label_tensor, rgb_tensor, normal_pred_tensor, normal_labels_tensor, pred_coords_tensor, \
            coords_tensor, pred_sem_seg_tensor, sem_masks_tensor), dim=3)

    # grid_image = make_grid(images, 1, normalize=True, scale_each=True)
    grid_image = make_grid(images, 1, normalize=False, scale_each=False)

    return grid_image
