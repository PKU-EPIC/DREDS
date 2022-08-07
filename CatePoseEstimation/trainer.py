import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from numpy.core.fromnumeric import put
from numpy.lib.twodim_base import mask_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils.loss_functions import loss_fn_l1, loss_fn_cosine, GradientLoss, weight_reduce_loss
from datasets.datasets import SwinDRNetDataset, augs_train, augs_test, input_only
from utils.create_image_grid import create_grid_image
from utils.metrics_depth_restoration import get_metrics_depth_restoration_train, get_metrics_depth_restoration_inference
from utils.metrics_sem_seg import get_metrics_sem_seg
from utils.api_utils import get_surface_normal
import cv2
import matplotlib.pyplot as plt
import time
import copy
import math
import _pickle as cPickle
from utils.metrics_nocs import align , draw_detections , compute_degree_cm_mAP ,prepare_data_posefitting

def depth_to_xyz(depthImage, fx, fy, scale_h=1., scale_w=1.):
    # input depth image[B, 1, H, W]
    # output xyz image[B, 3, H, W]
    fx = fx * scale_w
    fy = fy * scale_h
    B, C, H, W = depthImage.shape
    device = depthImage.device
    du = W//2 - 0.5
    dv = H//2 - 0.5

    xyz = torch.zeros([B, H, W, 3], device=device)
    imageIndexX = torch.arange(0, W, 1, device=device) - du
    imageIndexY = torch.arange(0, H, 1, device=device) - dv
    depthImage = depthImage.squeeze()
    if B == 1:
        depthImage = depthImage.unsqueeze(0)

    xyz[:, :, :, 0] = depthImage/fx * imageIndexX
    xyz[:, :, :, 1] = (depthImage.transpose(1, 2)/fy * imageIndexY.T).transpose(1, 2)
    xyz[:, :, :, 2] = depthImage
    xyz = xyz.permute(0, 3, 1, 2).to(device)
    return xyz

def smoothl1_diff(y_true,y_pred,threshold=0.1):
    diff = torch.abs (torch.sub(input=y_true, alpha=1, other=y_pred))
    coefficient = 1 / (2 * threshold)
    less = torch.pow(diff,2) * coefficient 
    more = diff - threshold/2
    loss = torch.where(diff<threshold,less,more)
    return loss 

def l1_loss (y_true,y_pred) :
    diff = torch.abs (torch.sub(input=y_true, alpha=1, other=y_pred))
    return diff 

def l2_loss (y_true,y_pred):  
    diff = torch.abs (torch.sub(input=y_true, alpha=1, other=y_pred))
    loss = torch.pow(diff,2)
    return loss

class SwinDRNetTrainer():
    def __init__(self, args, model, device_list, continue_ckpt_path):
        if continue_ckpt_path is not None:
            msg = model.load_state_dict(torch.load(continue_ckpt_path)['model_state_dict'])
            print("self trained swin unet", msg)
        
        self.material_mask = {'transparent': args.mask_transparent,
                              'specular': args.mask_specular,
                              'diffuse': args.mask_diffuse}

        self.val_interation_interval = args.val_interation_interval
        self.data_path_train = args.train_data_path
        self.data_path_val = args.val_data_path
        
        if args.val_depth_path :
            self.val_depth_path = args.val_depth_path
        else :
            self.val_depth_path = args.val_data_path
        self.output_path = args.output_dir
        self.obj_path_train = args.train_obj_path
        if args.val_obj_path :
            self.obj_path_val = args.val_obj_path
        else :
            self.val_depth_path = args.train_obj_path
        self.percentageDataForTraining = args.percentageDataForTraining
        self.percentageDataForVal = args.percentageDataForVal

        self.base_lr = args.base_lr
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size * args.n_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.val_data_type = args.val_data_type

        # realsense D415 
        self.fx_real_input = 918.295227050781           
        self.fy_real_input = 917.5439453125             
        self.fx_real_label = 918.295227050781 / 2.0
        self.fy_real_label = 917.5439453125 / 2.0

        # simulated depth sensor
        self.fx_sim = 446.31
        self.fy_sim = 446.31

        # the shape of depth map for computing metrics
        self.get_metrics_w = 224
        self.get_metrics_h = 126
        
        self.train_loader, self.val_loader = self.prepare_data()
        if args.n_gpu > 1:
            self.model = nn.DataParallel(model, device_list)
        else:
            self.model = model
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.base_lr, betas=(0.9, 0.999), weight_decay=0.01)
        if continue_ckpt_path is not None:
            self.optimizer.load_state_dict(torch.load(continue_ckpt_path)['optimizer_state_dict'])
            self.epoch = torch.load(continue_ckpt_path)['epoch']
            self.total_iter_num = self.epoch * len(self.train_loader)
        else:
            self.epoch = 0
            self.total_iter_num = 0
            
        self.writer = SummaryWriter(self.output_path + '/log')
        self.max_epochs = args.max_epochs
        self.max_iterations = self.max_epochs * len(self.train_loader) + 1
        
        logging.info("{} iterations per epoch. {} max iterations ".format(len(self.train_loader), self.max_iterations))
        if continue_ckpt_path is not None:
            self.iterator = tqdm(range(self.epoch + 1, self.max_epochs), ncols=70)
        else:
            self.iterator = tqdm(range(self.epoch, self.max_epochs), ncols=70)
            
        self.depth_restoration_train_with_masks = True   

        self.CHECKPOINT_DIR = args.checkpoint_save_path
        if os.path.exists(self.CHECKPOINT_DIR)==False:
            os.makedirs(self.CHECKPOINT_DIR)


        ################################### depth restoration metrics #########################################
        # depth restoration metrics: a1, a2, a3, rmse, abs_rel, mae        
        self.metrics_train_epoch_depth_restoration = {'total': [0.0 for i in range(6)], 
                                                      'instance': [0.0 for i in range(6)],
                                                      'background': [0.0 for i in range(6)]}

        self.metrics_val_epoch_depth_restoration = {'total': [0.0 for i in range(6)], 
                                                      'instance': [0.0 for i in range(6)],
                                                      'background': [0.0 for i in range(6)]}

        # semantic segmentation metrics: acc, all_acc, iou
        self.metrics_train_epoch_sem_seg = [0.0 for i in range(3)]
        self.metrics_val_epoch_sem_seg = [0.0 for i in range(3)]


        self.loss_val_epoch_depth_restoration = {'total': 0.0, 'instance': 0.0, 'background':0.0}  
        self.loss_val_iter_depth_restoration = {'total': 0.0, 'instance': 0.0, 'background':0.0}    
        self.loss_train_epoch_depth_restoration = {'total': 0.0, 'instance': 0.0, 'background':0.0}
        self.loss_train_iter_depth_restoration = {'total': 0.0, 'instance': 0.0, 'background':0.0}

        self.loss_val_epoch_sem_seg = 0.0
        self.loss_val_iter_sem_seg = 0.0
        self.loss_train_epoch_sem_seg = 0.0
        self.loss_train_iter_sem_seg = 0.0

        self.init_coord_loss()

        self.loss_val_epoch = 0.0
        self.loss_val_iter = 0.0
        self.loss_train_epoch = 0.0
        self.loss_train_iter = 0.0

        # from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
        logging.basicConfig(filename=self.output_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        print(self.data_path_train)
       
         
    def prepare_data(self):
        # generate trainLoader
        datasets_train = [
            {'rgb': self.data_path_train,
            'sim_depth': self.data_path_train,
            'output_depth' :self.data_path_train ,
            'syn_depth': self.data_path_train,
            'nocs': self.data_path_train,
            'mask': self.data_path_train,
            'meta': self.data_path_train,
            'obj':self.obj_path_train}
        ]
        
        db_synthetic_lst = []
        for dataset in datasets_train:                                      
            db_synthetic = SwinDRNetDataset(fx=self.fx_sim,
                                       fy=self.fy_sim,
                                       rgb_dir=dataset["rgb"],
                                       sim_depth_dir=dataset["sim_depth"],
                                       output_depth_dir=dataset["output_depth"] ,
                                       syn_depth_dir=dataset["syn_depth"],
                                       nocs_dir=dataset["nocs"],
                                       mask_dir=dataset["mask"],
                                       meta_dir=dataset["meta"],                                   
                                       obj_dir=dataset["obj"],   
                                       transform=augs_train,
                                       input_only=input_only,
                                       material_valid=self.material_mask)   
            db_synthetic_lst.append(db_synthetic)
        db_synthetic = torch.utils.data.ConcatDataset(db_synthetic_lst)

        train_size_synthetic = int(self.percentageDataForTraining * len(db_synthetic))
        db_train, _ = torch.utils.data.random_split(db_synthetic,
                                                (train_size_synthetic, len(db_synthetic) - train_size_synthetic))
        
        train_loader = DataLoader(db_train,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True,
                                  pin_memory=True)


        # generate validationLoader
        datasets_val = [
            {'rgb': self.data_path_val,
            'sim_depth': self.data_path_val,
            'output_depth' :self.val_depth_path ,
            'syn_depth': self.data_path_val,
            'nocs': self.data_path_val,
            'mask': self.data_path_val,
            'meta': self.data_path_val,
            'obj':self.obj_path_val}
        ]
        db_val_list = []
        for dataset in datasets_val:
            if self.val_data_type == 'real':
                db_val = SwinDRNetDataset(fx=self.fx_real_input,
                                    fy=self.fy_real_input,
                                    rgb_dir=dataset["rgb"],
                                    sim_depth_dir=dataset["sim_depth"],
                                    output_depth_dir=dataset["output_depth"] ,
                                    syn_depth_dir=dataset["syn_depth"],
                                    nocs_dir=dataset["nocs"],
                                    mask_dir=dataset["mask"],
                                    meta_dir=dataset["meta"],   
                                    obj_dir=dataset["obj"],   
                                    transform=augs_test,
                                    input_only=None,
                                    material_valid=self.material_mask)
            else:
                db_val = SwinDRNetDataset(fx=self.fx_sim,
                                    fy=self.fy_sim,
                                    rgb_dir=dataset["rgb"],
                                    sim_depth_dir=dataset["sim_depth"],
                                    output_depth_dir=dataset["output_depth"] ,
                                    syn_depth_dir=dataset["syn_depth"],
                                    nocs_dir=dataset["nocs"],
                                    mask_dir=dataset["mask"],
                                    meta_dir=dataset["meta"],   
                                    obj_dir=dataset["obj"],   
                                    transform=augs_test,
                                    input_only=None,
                                    material_valid=self.material_mask)            
            db_val_list.append(db_val)
        
        db_val = torch.utils.data.ConcatDataset(db_val_list)
        # if self.percentageDataForVal != 1:
        val_size = int(self.percentageDataForVal * len(db_val))
        db_val, _ = torch.utils.data.random_split(db_val,
                                                (val_size, len(db_val) - val_size))
        val_loader = DataLoader(db_val,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=8,
                                drop_last=True,
                                pin_memory=True)

        return train_loader, val_loader

    def init_coord_loss(self):
        self.loss_val_epoch_coord = 0.0
        self.loss_val_iter_coord = 0.0
        self.loss_train_epoch_coord = 0.0
        self.loss_train_iter_coord = 0.0
        
        self.loss_val_epoch_coord_x_loss = 0.0
        self.loss_val_iter_coord_x_loss = 0.0
        self.loss_train_epoch_coord_x_loss = 0.0
        self.loss_train_iter_coord_x_loss = 0.0

        self.loss_val_epoch_coord_y_loss = 0.0
        self.loss_val_iter_coord_y_loss = 0.0
        self.loss_train_epoch_coord_y_loss = 0.0
        self.loss_train_iter_coord_y_loss = 0.0
        
        self.loss_val_epoch_coord_z_loss = 0.0
        self.loss_val_iter_coord_z_loss = 0.0
        self.loss_train_epoch_coord_z_loss = 0.0
        self.loss_train_iter_coord_z_loss = 0.0

        self.loss_val_epoch_coord_l1_diff = 0.0
        self.loss_val_iter_coord_l1_diff = 0.0
        self.loss_train_epoch_coord_l1_diff = 0.0
        self.loss_train_iter_coord_l1_diff = 0.0

        self.loss_val_epoch_coord_l2_diff = 0.0
        self.loss_val_iter_coord_l2_diff = 0.0
        self.loss_train_epoch_coord_l2_diff = 0.0
        self.loss_train_iter_coord_l2_diff = 0.0
    

    def transfer_to_device(self, sample_batched):
        for key in sample_batched:
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def transfer_to_cpu(self, sample_batched):
        for key in sample_batched:
            sample_batched[key] = sample_batched[key].to('cpu')
        return sample_batched

    def get_loss_depth_restoration(self, syn_ds, ins_masks, pred_ds, pred_ds_initial, mode):
        # f = 502.3
        f = 446.31
        ori_h = 360
        ori_w = 640

        loss_instance_weight = 1.0
        loss_background_weight = 0.4
        loss_normal_weight = 0.1
        loss_grad_weight = 0.6
        loss_weight_d_initial = 0.4

        l1 = nn.L1Loss(reduction='none')
        huber_loss_fn = nn.SmoothL1Loss(reduction='none')
        grad_loss = GradientLoss()

        normal_labels, _, _ = get_surface_normal(syn_ds, f, syn_ds.shape[2] / ori_h, syn_ds.shape[3] / ori_w)
        normal_pred, _, _ = get_surface_normal(pred_ds, f, pred_ds.shape[2] / ori_h, pred_ds.shape[3] / ori_w)
        normal_pred_ori, _, _ = get_surface_normal(pred_ds_initial, f, pred_ds.shape[2] / ori_h, pred_ds.shape[3] / ori_w)

        loss_d_initial = l1(pred_ds_initial, syn_ds) + \
                loss_normal_weight * torch.mean(l1(normal_pred_ori, normal_labels), dim=1, keepdim=True) + \
                loss_grad_weight * grad_loss(pred_ds_initial, syn_ds)
        
        loss_d_with_confidence = l1(pred_ds, syn_ds) + \
                loss_normal_weight * torch.mean(l1(normal_pred, normal_labels), dim=1, keepdim=True) + \
                loss_grad_weight * grad_loss(pred_ds, syn_ds)

        loss = loss_d_with_confidence + loss_weight_d_initial * loss_d_initial
        
        if not self.depth_restoration_train_with_masks:
            loss = loss.mean()
        else:
            num_instance = torch.sum(ins_masks)
            num_background = torch.sum(1 - ins_masks)
            loss_instance = torch.sum(loss * ins_masks) / num_instance
            loss_background = torch.sum(loss * (1 - ins_masks)) / num_background
            loss = loss_instance_weight * loss_instance + loss_background_weight * loss_background
            if mode == 'train':
                self.loss_train_iter_depth_restoration['instance'] = loss_instance
                self.loss_train_iter_depth_restoration['background'] = loss_background
            else:
                self.loss_val_iter_depth_restoration['instance'] = loss_instance
                self.loss_val_iter_depth_restoration['background'] = loss_background

        if mode == 'train':
            self.loss_train_iter_depth_restoration['total'] = loss
        else:
            self.loss_val_iter_depth_restoration['total'] = loss 

        return normal_labels, normal_pred


    def get_loss_sem_seg(self, sem_masks, pred_sem_seg, mode):
        loss = F.cross_entropy(pred_sem_seg,
                               sem_masks.squeeze(1).long(),
                               weight=None,
                               reduction='none',
                               ignore_index=255)
        loss = weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None)
        if mode == 'train':
            self.loss_train_iter_sem_seg = loss
        else:
            self.loss_val_iter_sem_seg = loss


    def get_loss_coord(self, coords, pred_coords, masks, mode):
        l1 = nn.L1Loss(reduction='none')
        loss = l1(pred_coords, coords)
        num_instance = torch.sum(masks)
        loss = torch.sum(loss * masks) / num_instance

        if mode == 'train':
            self.loss_train_iter_coord = loss
        else:
            self.loss_val_iter_coord = loss

    def mrcnn_coords_symmetry_loss_graph_3(self, coords, pred_coords, masks,sem_masks,loss_fn, mode):

        coords = coords.permute(0,2,3,1)
        pred_coords = pred_coords.permute(0,2,3,1)
        batch ,height,width ,_= coords.shape
        sym_masks = sem_masks.clone().detach()
        nosym_masks = sem_masks.clone().detach()
        sym_masks[(sym_masks==3) | (sym_masks==5) | (sym_masks==6) | (sym_masks==7) |(sym_masks==8)]=0
        sym_masks[sym_masks!=0] = 1
        sym_masks = sym_masks.permute(0,2,3,1)
         
        nosym_masks[(nosym_masks==1) | (nosym_masks==2) | (nosym_masks==4)|(nosym_masks==8) ]=0
        nosym_masks[nosym_masks!=0] = 1
        nosym_masks = nosym_masks.permute(0,2,3,1)

        pos_class_rotation_theta = torch.tensor(2*math.pi/6,dtype=torch.float64)
        pos_class_cos = torch.cos(pos_class_rotation_theta)
        pos_class_sin = torch.sin(pos_class_rotation_theta)
        pos_class_one = torch.ones(pos_class_rotation_theta.shape)
        pos_class_zero = torch.zeros(pos_class_rotation_theta.shape)
        line_1 = torch.stack((pos_class_cos,pos_class_zero,pos_class_sin))
        line_2 = torch.stack((pos_class_zero,pos_class_one,pos_class_zero))
        line_3 = torch.stack((pos_class_sin.neg(),pos_class_zero,pos_class_cos))
        pos_class_rotation_matrix = torch.stack((line_1,line_2,line_3),dim=1)
        tiled_rotation_matrix=pos_class_rotation_matrix.view(1, 1, 1, 3, 3).repeat(batch,height,width,1,1).cuda()


        y_true = coords - 0.5
        y_true = y_true.unsqueeze(4)  ## shape: [batch, height, width, 3, 1]

        rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
        rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
        rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
        rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
        rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
        y_true_stack = torch.cat((y_true, rotated_y_true_1, rotated_y_true_2, rotated_y_true_3,
                        rotated_y_true_4, rotated_y_true_5),dim=4) ## shape: [batch, height, width, 3, 6]

        y_true_stack = y_true_stack.permute(0, 1, 2, 4, 3)  ## shape: [batch, height, width, 6, 3]
        y_true_stack = y_true_stack + 0.5
        
        y_pred = pred_coords.unsqueeze(3) 
        y_pred_stack = y_pred.repeat(1, 1, 1,   y_true_stack.shape[3], 1 )
        if loss_fn == 'Soft_L1':
            fn = smoothl1_diff
        elif loss_fn == 'L1':
            fn = l1_loss  
        elif loss_fn == 'L2':
            fn = l2_loss
        loss_sym = fn(y_true_stack,y_pred_stack)
        
        reshape_sym_masks = sym_masks.view(sym_masks.shape[0], sym_masks.shape[1], sym_masks.shape[2],  1, 1)


        sym_loss_in_mask = loss_sym.mul(reshape_sym_masks)
        sum_sym_loss_in_mask =sym_loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois,6,  3]
        total_sum_sym_loss_in_mask = sum_sym_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
        arg_min_rotation = torch.argmin(total_sum_sym_loss_in_mask, dim=-1).long()  ##shape: [num_pos_rois]
        index_1 = torch.arange(0,arg_min_rotation.shape[0]).cuda().long()
        min_sym_loss_in_mask = sum_sym_loss_in_mask[index_1, arg_min_rotation, :]  ## shape: [num_pos_rois, 3]

    
        loss_nosym = fn(pred_coords, coords)
        reshape_nosym_masks = nosym_masks.view(nosym_masks.shape[0], nosym_masks.shape[1], nosym_masks.shape[2],   1)
        nosym_loss_in_mask = loss_nosym.mul(reshape_nosym_masks)
        sum_nosym_loss_in_mask =nosym_loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois,  3]

        loss = min_sym_loss_in_mask+sum_nosym_loss_in_mask
        num_of_pixels = masks.sum(dim = [1, 2,3]) + 0.00001  ## shape: [num_pos_rois]
        num_of_pixels = num_of_pixels.unsqueeze(1)

        mean_loss_in_mask = loss.div(num_of_pixels.expand_as(loss))  ## shape: [num_pos_rois, 3]
        final_loss = mean_loss_in_mask.mean(dim=0)
        if loss == "L2":
            final_loss = torch.sqrt(final_loss)
        

        if mode == 'train':
            #self.loss_train_iter_coord = final_loss
            self.loss_train_iter_coord_x_loss , self.loss_train_iter_coord_y_loss ,self.loss_train_iter_coord_z_loss = final_loss
        else:
            self.loss_val_iter_coord_x_loss , self.loss_val_iter_coord_y_loss , self.loss_val_iter_coord_z_loss= final_loss

    def mrcnn_coords_symmetry_loss_graph_1(self, coords, pred_coords, masks,sem_masks,loss_fn, mode):

        coords = coords.permute(0,2,3,1)
        pred_coords = pred_coords.permute(0,2,3,1)
        batch ,height,width ,_= coords.shape
        sym_masks = sem_masks.clone().detach()
        nosym_masks = sem_masks.clone().detach()
        sym_masks[(sym_masks==3) | (sym_masks==5) | (sym_masks==6) | (sym_masks==7) |(sym_masks==8)]=0
        sym_masks[sym_masks!=0] = 1
        sym_masks = sym_masks.permute(0,2,3,1)
         
        nosym_masks[(nosym_masks==1) | (nosym_masks==2) | (nosym_masks==4)|(nosym_masks==8) ]=0
        nosym_masks[nosym_masks!=0] = 1
        nosym_masks = nosym_masks.permute(0,2,3,1)

        pos_class_rotation_theta = torch.tensor(2*math.pi/6,dtype=torch.float64)
        pos_class_cos = torch.cos(pos_class_rotation_theta)
        pos_class_sin = torch.sin(pos_class_rotation_theta)
        pos_class_one = torch.ones(pos_class_rotation_theta.shape)
        pos_class_zero = torch.zeros(pos_class_rotation_theta.shape)
        line_1 = torch.stack((pos_class_cos,pos_class_zero,pos_class_sin))
        line_2 = torch.stack((pos_class_zero,pos_class_one,pos_class_zero))
        line_3 = torch.stack((pos_class_sin.neg(),pos_class_zero,pos_class_cos))
        pos_class_rotation_matrix = torch.stack((line_1,line_2,line_3),dim=1)
        tiled_rotation_matrix=pos_class_rotation_matrix.view(1, 1, 1, 3, 3).repeat(batch,height,width,1,1).cuda()


        y_true = coords - 0.5
        y_true = y_true.unsqueeze(4)  ## shape: [batch, height, width, 3, 1]

        rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
        rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
        rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
        rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
        rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
        y_true_stack = torch.cat((y_true, rotated_y_true_1, rotated_y_true_2, rotated_y_true_3,
                        rotated_y_true_4, rotated_y_true_5),dim=4) ## shape: [batch, height, width, 3, 6]

        y_true_stack = y_true_stack.permute(0, 1, 2, 4, 3)  ## shape: [batch, height, width, 6, 3]
        y_true_stack = y_true_stack + 0.5
        
        y_pred = pred_coords.unsqueeze(3) 
        y_pred_stack = y_pred.repeat(1, 1, 1,   y_true_stack.shape[3], 1 )

        if loss_fn == 'L1':
            fn = l1_loss  
        elif loss_fn == 'L2':
            fn = l2_loss
        loss_sym = fn(y_true_stack,y_pred_stack)
        
        reshape_sym_masks = sym_masks.view(sym_masks.shape[0], sym_masks.shape[1], sym_masks.shape[2],  1, 1)
        sym_loss_in_mask = loss_sym.mul(reshape_sym_masks)
        sum_sym_loss_in_mask =sym_loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois,6,  3]
        total_sum_sym_loss_in_mask = sum_sym_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
        arg_min_rotation = torch.argmin(total_sum_sym_loss_in_mask, dim=-1).long()  ##shape: [num_pos_rois]
        index_1 = torch.arange(0,arg_min_rotation.shape[0]).cuda().long()
        min_sym_loss_in_mask = sum_sym_loss_in_mask[index_1, arg_min_rotation, :]  ## shape: [num_pos_rois, 3]

    
        loss_nosym = fn(pred_coords, coords)
        reshape_nosym_masks = nosym_masks.view(nosym_masks.shape[0], nosym_masks.shape[1], nosym_masks.shape[2],   1)
        nosym_loss_in_mask = loss_nosym.mul(reshape_nosym_masks)
        sum_nosym_loss_in_mask =nosym_loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois,  3]

        loss = min_sym_loss_in_mask+sum_nosym_loss_in_mask
        num_of_pixels = masks.sum(dim = [1, 2,3]) + 0.00001  ## shape: [num_pos_rois]
        num_of_pixels = num_of_pixels.unsqueeze(1)

        mean_loss_in_mask = loss.div(num_of_pixels.expand_as(loss))  ## shape: [num_pos_rois, 3]
        final_loss = mean_loss_in_mask.mean(dim=0)
        if loss == "L2":
            final_loss = torch.sqrt(final_loss)
        final_loss =final_loss.mean(dim=0)

        if loss_fn == 'L1':
            final_loss =final_loss.mean(dim=0) 
            if mode == 'train':
                #self.loss_train_iter_coord = final_loss
                self.loss_train_iter_coord_l1_diff = final_loss
            else:
                self.loss_val_iter_coord_l1_diff= final_loss
        elif loss_fn == 'L2':
            final_loss = torch.sqrt(final_loss)
            final_loss =final_loss.mean(dim=0)
            if mode == 'train':
                #self.loss_train_iter_coord = final_loss
                self.loss_train_iter_coord_l2_diff = final_loss
            else:
                self.loss_val_iter_coord_l2_diff= final_loss
    

    def forward(self, sample_batched, mode):
        loss_weight_d = 1.0
        loss_weight_sem_seg = 1.0
        loss_weight_coord = 3.0

        rgbs = sample_batched['rgb']
        sim_xyzs = sample_batched['sim_xyz']
        sim_ds = sample_batched['sim_depth']
        syn_ds = sample_batched['syn_depth']
        coords = sample_batched['nocs_map']
        sem_masks = sample_batched['sem_mask']
        ins_masks = sample_batched['ins_mask']
        ins_without_other_masks = sample_batched['ins_w/o_others_mask']

        # pred_ds, pred_ds_initial, confidence_sim_ds, confidence_initial, pred_sem_seg, pred_coords = self.model(rgbs, sim_ds)    # [bs, 150, 512, 512], [bs, 150, 512, 512])
        pred_sem_seg, pred_coords  = self.model(rgbs, sim_xyzs)    # [bs, 150, 512, 512], [bs, 150, 512, 512])
        self.get_loss_sem_seg(sem_masks, pred_sem_seg, mode)

        self.mrcnn_coords_symmetry_loss_graph_3(coords, pred_coords, ins_without_other_masks,sem_masks,'Soft_L1', mode)
        self.mrcnn_coords_symmetry_loss_graph_1(coords, pred_coords, ins_without_other_masks,sem_masks,'L1', mode)
        self.mrcnn_coords_symmetry_loss_graph_1(coords, pred_coords, ins_without_other_masks,sem_masks,'L2', mode)
        if mode == 'train':
            self.loss_train_iter = loss_weight_coord * (self.loss_train_iter_coord_x_loss + self.loss_train_iter_coord_y_loss +\
                                    self.loss_train_iter_coord_z_loss  + self.loss_train_iter_coord_l2_diff) + \
                                   loss_weight_sem_seg * self.loss_train_iter_sem_seg 
        else:
            self.loss_val_iter = loss_weight_coord * (self.loss_val_iter_coord_x_loss + self.loss_val_iter_coord_y_loss +\
                                self.loss_val_iter_coord_z_loss  + self.loss_val_iter_coord_l2_diff+ self.loss_val_iter_coord_l1_diff) +\
                                 loss_weight_sem_seg * self.loss_val_iter_sem_seg
        
        # return pred_ds, normal_labels, normal_pred, confidence_sim_ds, confidence_initial, pred_sem_seg, pred_coords
        return pred_sem_seg, pred_coords
    

    def save_iter_depth_restoration_metrics(self, labels, outputs, masks, mode):
        a1, a2, a3, rmse, abs_rel, mae = get_metrics_depth_restoration_train(labels, outputs, self.get_metrics_w, self.get_metrics_h)  
        num_samples_train = len(self.train_loader)
        num_samples_val = len(self.val_loader)
        if mode == 'train':
            ############################### depth restoration loss #########################################
            self.loss_train_epoch_depth_restoration['total'] += self.loss_train_iter_depth_restoration['total'].item() / num_samples_train
            self.writer.add_scalar('Train/loss/Train_Loss', self.loss_train_iter_depth_restoration['total'].item(), self.total_iter_num)

            ############################### depth restoration metrics ######################################
            self.metrics_train_epoch_depth_restoration['total'][0] += a1.item() / num_samples_train
            self.metrics_train_epoch_depth_restoration['total'][1] += a2.item() / num_samples_train
            self.metrics_train_epoch_depth_restoration['total'][2] += a3.item() / num_samples_train
            self.metrics_train_epoch_depth_restoration['total'][3] += rmse.item() / num_samples_train
            self.metrics_train_epoch_depth_restoration['total'][4] += abs_rel.item() / num_samples_train
            self.metrics_train_epoch_depth_restoration['total'][5] += mae.item() / num_samples_train

            self.writer.add_scalar('Train/metrics/Train_a1', a1.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_a2', a2.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_a3', a3.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_rmse', rmse.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_abs_rel', abs_rel.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_mae', mae.item(), self.total_iter_num)


            if self.depth_restoration_train_with_masks:
                ############################### depth restoration loss #########################################
                self.loss_train_epoch_depth_restoration['instance'] += self.loss_train_iter_depth_restoration['instance'].item() / num_samples_train
                self.loss_train_epoch_depth_restoration['background'] += self.loss_train_iter_depth_restoration['background'].item() / num_samples_train
                self.writer.add_scalar('Train/loss/Train_Instance_Loss', self.loss_train_iter_depth_restoration['instance'].item(), self.total_iter_num)
                self.writer.add_scalar('Train/loss/Train_Background_Loss', self.loss_train_iter_depth_restoration['background'].item(), self.total_iter_num)
                
                ############################### depth restoration metrics ######################################
                # instance metrics
                instance_a1, instance_a2, instance_a3, instance_rmse, instance_abs_rel, instance_mae = get_metrics_depth_restoration_train(labels, outputs, self.get_metrics_w, self.get_metrics_h, masks)   
                self.metrics_train_epoch_depth_restoration['instance'][0] += instance_a1.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['instance'][1] += instance_a2.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['instance'][2] += instance_a3.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['instance'][3] += instance_rmse.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['instance'][4] += instance_abs_rel.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['instance'][5] += instance_mae.item() / num_samples_train

                # background metrics
                background_a1, background_a2, background_a3, background_rmse, background_abs_rel, background_mae = get_metrics_depth_restoration_train(labels, outputs, self.get_metrics_w, self.get_metrics_h, 1 - masks)   
                self.metrics_train_epoch_depth_restoration['background'][0] += background_a1.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['background'][1] += background_a2.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['background'][2] += background_a3.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['background'][3] += background_rmse.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['background'][4] += background_abs_rel.item() / num_samples_train
                self.metrics_train_epoch_depth_restoration['background'][5] += background_mae.item() / num_samples_train    

                self.writer.add_scalar('Train/metrics/Instance/Train_a1', instance_a1.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Instance/Train_a2', instance_a2.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Instance/Train_a3', instance_a3.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Instance/Train_rmse', instance_rmse.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Instance/Train_abs_rel', instance_abs_rel.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Instance/Train_mae', instance_mae.item(), self.total_iter_num)

                self.writer.add_scalar('Train/metrics/Background/Train_a1', background_a1.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Background/Train_a2', background_a2.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Background/Train_a3', background_a3.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Background/Train_rmse', background_rmse.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Background/Train_abs_rel', background_abs_rel.item(), self.total_iter_num)
                self.writer.add_scalar('Train/metrics/Background/Train_mae', background_mae.item(), self.total_iter_num)

        else:
            ############################### depth restoration loss #########################################
            self.loss_val_epoch_depth_restoration['total'] += self.loss_val_iter_depth_restoration['total'].item() / num_samples_val

            ############################### depth restoration metrics ######################################
            self.metrics_val_epoch_depth_restoration['total'][0] += a1.item() / num_samples_val
            self.metrics_val_epoch_depth_restoration['total'][1] += a2.item() / num_samples_val
            self.metrics_val_epoch_depth_restoration['total'][2] += a3.item() / num_samples_val
            self.metrics_val_epoch_depth_restoration['total'][3] += rmse.item() / num_samples_val
            self.metrics_val_epoch_depth_restoration['total'][4] += abs_rel.item() / num_samples_val
            self.metrics_val_epoch_depth_restoration['total'][5] += mae.item() / num_samples_val

            if self.depth_restoration_train_with_masks:
                ############################### depth restoration loss #########################################
                self.loss_val_epoch_depth_restoration['instance'] += self.loss_val_iter_depth_restoration['instance'].item() / num_samples_val
                self.loss_val_epoch_depth_restoration['background'] += self.loss_val_iter_depth_restoration['background'].item() / num_samples_val
  
                ############################### depth restoration metrics ######################################
                # instance metrics
                instance_a1, instance_a2, instance_a3, instance_rmse, instance_abs_rel, instance_mae = get_metrics_depth_restoration_train(labels, outputs, self.get_metrics_w, self.get_metrics_h, masks)   
                self.metrics_val_epoch_depth_restoration['instance'][0] += instance_a1.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['instance'][1] += instance_a2.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['instance'][2] += instance_a3.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['instance'][3] += instance_rmse.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['instance'][4] += instance_abs_rel.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['instance'][5] += instance_mae.item() / num_samples_val  

                # background metrics
                background_a1, background_a2, background_a3, background_rmse, background_abs_rel, background_mae = get_metrics_depth_restoration_train(labels, outputs, self.get_metrics_w, self.get_metrics_h, 1 - masks)   
                self.metrics_val_epoch_depth_restoration['background'][0] += background_a1.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['background'][1] += background_a2.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['background'][2] += background_a3.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['background'][3] += background_rmse.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['background'][4] += background_abs_rel.item() / num_samples_val
                self.metrics_val_epoch_depth_restoration['background'][5] += background_mae.item() / num_samples_val    


    def save_epoch_depth_restoration_metrics(self, mode='val'):
        # Log Val Loss
        self.writer.add_scalar('Val/loss/Val_Loss', self.loss_val_epoch_depth_restoration['total'], self.epoch)
        print('Val_Loss: {:.4f}'.format(self.loss_val_epoch_depth_restoration['total']))

        if self.depth_restoration_train_with_masks:
            if mode == 'val':
                self.writer.add_scalar('Val/loss/Val_Instance_Loss', self.loss_val_epoch_depth_restoration['instance'], self.epoch)
                self.writer.add_scalar('Val/loss/Val_Background_Loss', self.loss_val_epoch_depth_restoration['background'], self.epoch)
            print('Val_Instance_Loss: {:.4f}'.format(self.loss_val_epoch_depth_restoration['instance']))
            print('Val_Background_Loss: {:.4f}'.format(self.loss_val_epoch_depth_restoration['background']))

        if mode == 'val':
            self.writer.add_scalar('Val/metrics/Val_a1', self.metrics_val_epoch_depth_restoration['total'][0], self.epoch)
            self.writer.add_scalar('Val/metrics/Val_a2', self.metrics_val_epoch_depth_restoration['total'][1], self.epoch)
            self.writer.add_scalar('Val/metrics/Val_a3', self.metrics_val_epoch_depth_restoration['total'][2], self.epoch)
            self.writer.add_scalar('Val/metrics/Val_rmse', self.metrics_val_epoch_depth_restoration['total'][3], self.epoch)
            self.writer.add_scalar('Val/metrics/Val_abs_rel', self.metrics_val_epoch_depth_restoration['total'][4], self.epoch)
            self.writer.add_scalar('Val/metrics/Val_mae', self.metrics_val_epoch_depth_restoration['total'][5], self.epoch)

        print('Val_a1: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['total'][0]))
        print('Val_a2: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['total'][1]))
        print('Val_a3: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['total'][2]))
        print('Val_rmse: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['total'][3]))
        print('Val_rel: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['total'][4]))
        print('Val_mae: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['total'][5]))

        if self.depth_restoration_train_with_masks:
            if mode == 'val':
                self.writer.add_scalar('Val/metrics/Instance/Val_a1', self.metrics_val_epoch_depth_restoration['instance'][0], self.epoch)
                self.writer.add_scalar('Val/metrics/Instance/Val_a2', self.metrics_val_epoch_depth_restoration['instance'][1], self.epoch)
                self.writer.add_scalar('Val/metrics/Instance/Val_a3', self.metrics_val_epoch_depth_restoration['instance'][2], self.epoch)
                self.writer.add_scalar('Val/metrics/Instance/Val_rmse', self.metrics_val_epoch_depth_restoration['instance'][3], self.epoch)
                self.writer.add_scalar('Val/metrics/Instance/Val_abs_rel', self.metrics_val_epoch_depth_restoration['instance'][4], self.epoch)
                self.writer.add_scalar('Val/metrics/Instance/Val_mae', self.metrics_val_epoch_depth_restoration['instance'][5], self.epoch)

            print('Val_instance_a1: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['instance'][0]))
            print('Val_instance_a2: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['instance'][1]))
            print('Val_instance_a3: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['instance'][2]))
            print('Val_instance_rmse: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['instance'][3]))
            print('Val_instance_rel: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['instance'][4]))
            print('Val_instance_mae: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['instance'][5]))

            if mode == 'val':
                self.writer.add_scalar('Val/metrics/Background/Val_a1', self.metrics_val_epoch_depth_restoration['background'][0], self.epoch)
                self.writer.add_scalar('Val/metrics/Background/Val_a2', self.metrics_val_epoch_depth_restoration['background'][1], self.epoch)
                self.writer.add_scalar('Val/metrics/Background/Val_a3', self.metrics_val_epoch_depth_restoration['background'][2], self.epoch)
                self.writer.add_scalar('Val/metrics/Background/Val_rmse', self.metrics_val_epoch_depth_restoration['background'][3], self.epoch)
                self.writer.add_scalar('Val/metrics/Background/Val_abs_rel', self.metrics_val_epoch_depth_restoration['background'][4], self.epoch)
                self.writer.add_scalar('Val/metrics/Background/Val_mae', self.metrics_val_epoch_depth_restoration['background'][5], self.epoch)

            print('Val_background_a1: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['background'][0]))
            print('Val_background_a2: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['background'][1]))
            print('Val_background_a3: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['background'][2]))
            print('Val_background_rmse: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['background'][3]))
            print('Val_background_rel: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['background'][4]))
            print('Val_background_mae: {:.4f}'.format(self.metrics_val_epoch_depth_restoration['background'][5]))

        # depth restoration metrics: a1, a2, a3, rmse, abs_rel, mae        
        self.metrics_train_epoch_depth_restoration = {'total': [0.0 for i in range(6)], 
                                                      'instance': [0.0 for i in range(6)],
                                                      'background': [0.0 for i in range(6)]}

        self.metrics_val_epoch_depth_restoration = {'total': [0.0 for i in range(6)], 
                                                      'instance': [0.0 for i in range(6)],
                                                      'background': [0.0 for i in range(6)]}

        self.loss_train_epoch_depth_restoration = {'total': 0.0,
                                                   'instance': 0.0,
                                                   'background':0.0}
        self.loss_train_iter_depth_restoration = {'total': 0.0,
                                                   'instance': 0.0,
                                                   'background':0.0}

        self.loss_val_epoch_depth_restoration = {'total': 0.0,
                                                   'instance': 0.0,
                                                   'background':0.0}  
        self.loss_val_iter_depth_restoration = {'total': 0.0,
                                                   'instance': 0.0,
                                                   'background':0.0}                                                           


    def save_iter_sem_seg_metrics(self, labels, outputs, mode):
        # compute metrics
        num_samples_train = len(self.train_loader)
        num_samples_val = len(self.val_loader)

        outputs = F.softmax(outputs, dim=1)  # [1, 150, 512, 512]
        seg_pred = outputs.argmax(dim=1)  # [1, 512, 512]
        seg_pred = seg_pred.detach().cpu()
        outputs_list = list(seg_pred.clone().numpy())
        seg_masks = labels.squeeze(1)
        labels = seg_masks.detach().cpu()
        _labels = labels.numpy().copy()
        labels_list = list(_labels)  # labels_list[0].shape [512,512]
        ret_metrics = get_metrics_sem_seg(
            outputs_list,
            labels_list,
            self.num_classes,
            ignore_index=255,
            metrics=["mIoU"],
            label_map=dict(),
            # reduce_zero_label=True
            reduce_zero_label=False)
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        all_acc, acc, iou = ret_metrics_mean

        # statistics
        if mode == 'train':
            self.loss_train_epoch_sem_seg += self.loss_train_iter_sem_seg.item() / num_samples_train
            self.metrics_train_epoch_sem_seg[0] += acc / num_samples_train
            self.metrics_train_epoch_sem_seg[1] += all_acc / num_samples_train
            self.metrics_train_epoch_sem_seg[2] += iou / num_samples_train

            self.writer.add_scalar('Train/loss/Train_Loss_Sem_Seg', self.loss_train_iter_sem_seg.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_All_Acc', all_acc.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_Acc', acc.item(), self.total_iter_num)
            self.writer.add_scalar('Train/metrics/Train_IoU', iou.item(), self.total_iter_num)
        else:
            self.loss_val_epoch_sem_seg += self.loss_val_iter_sem_seg.item() / num_samples_val
            self.metrics_val_epoch_sem_seg[0] += acc / num_samples_val
            self.metrics_val_epoch_sem_seg[1] += all_acc / num_samples_val
            self.metrics_val_epoch_sem_seg[2] += iou / num_samples_val


    def save_epoch_sem_seg_metrics(self):
        # Log Val Loss
        self.writer.add_scalar('Val/loss/Val_Loss_Sem_Seg', self.loss_val_epoch_sem_seg, self.epoch)
        print('Val_Loss_Sem_Seg: {:.4f}'.format(self.loss_val_epoch_sem_seg))

        self.writer.add_scalar('Val/metrics/Val_Acc_All', self.metrics_val_epoch_sem_seg[0], self.epoch)
        self.writer.add_scalar('Val/metrics/Val_Acc', self.metrics_val_epoch_sem_seg[1], self.epoch)
        self.writer.add_scalar('Val/metrics/Val_IoU', self.metrics_val_epoch_sem_seg[2], self.epoch)

        print('Val_acc_all: {:.4f}'.format(self.metrics_val_epoch_sem_seg[0]))
        print('Val_acc: {:.4f}'.format(self.metrics_val_epoch_sem_seg[1]))
        print('Val_iou: {:.4f}'.format(self.metrics_val_epoch_sem_seg[2]))

        # semantic segmentation metrics: acc, all_acc, iou
        self.metrics_train_epoch_sem_seg = [0.0 for i in range(3)]
        self.metrics_val_epoch_sem_seg = [0.0 for i in range(3)]

        self.loss_val_epoch_sem_seg = 0.0
        self.loss_val_iter_sem_seg = 0.0
        self.loss_train_epoch_sem_seg = 0.0
        self.loss_train_iter_sem_seg = 0.0                                                    


    def save_iter_coord_metrics(self, mode):
        num_samples_train = len(self.train_loader)
        num_samples_val = len(self.val_loader)
        if mode == 'train':
            #self.loss_train_epoch_coord = self.loss_train_iter_coord_x_loss.item()+self.loss_train_iter_coord_y_loss.item()+\
            #                            self.loss_train_iter_coord_z_loss.item()+self.loss_train_iter_coord_l1_diff.item()+\
            #                            self.loss_train_iter_coord_l2_diff.item() + self.loss_train_epoch_coord
            #self.loss_train_epoch_coord = self.loss_train_epoch_coord / num_samples_train
            self.writer.add_scalar('Train/loss/Train_Loss_Coord_x', self.loss_train_iter_coord_x_loss.item(), self.total_iter_num)
            self.writer.add_scalar('Train/loss/Train_Loss_Coord_y', self.loss_train_iter_coord_y_loss.item(), self.total_iter_num)
            self.writer.add_scalar('Train/loss/Train_Loss_Coord_z', self.loss_train_iter_coord_z_loss.item(), self.total_iter_num)
            self.writer.add_scalar('Train/loss/Train_Loss_Coord_l1', self.loss_train_iter_coord_l1_diff.item(), self.total_iter_num)
            self.writer.add_scalar('Train/loss/Train_Loss_Coord_l2', self.loss_train_iter_coord_l2_diff.item(), self.total_iter_num)
        else:
            self.loss_val_epoch_coord_x_loss += self.loss_val_iter_coord_x_loss.item() / num_samples_val
            self.loss_val_epoch_coord_y_loss += self.loss_val_iter_coord_y_loss.item() / num_samples_val
            self.loss_val_epoch_coord_z_loss += self.loss_val_iter_coord_z_loss.item() / num_samples_val
            self.loss_val_epoch_coord_l1_diff += self.loss_val_iter_coord_l1_diff.item() / num_samples_val
            self.loss_val_epoch_coord_l2_diff += self.loss_val_iter_coord_l2_diff.item() / num_samples_val


    def save_epoch_coord_metrics(self):
        # Log Val Loss
        self.writer.add_scalar('Val/loss/Val_Loss_Coord_x', self.loss_val_epoch_coord_x_loss, self.total_iter_num)
        self.writer.add_scalar('Val/loss/Val_Loss_Coord_y', self.loss_val_epoch_coord_y_loss, self.total_iter_num)
        self.writer.add_scalar('Val/loss/Val_Loss_Coord_z', self.loss_val_epoch_coord_z_loss, self.total_iter_num)
        self.writer.add_scalar('Val/loss/Val_Loss_Coord_l1', self.loss_val_epoch_coord_l1_diff, self.total_iter_num)
        self.writer.add_scalar('Val/loss/Val_Loss_Coord_l2', self.loss_val_epoch_coord_l2_diff, self.total_iter_num)
        #print('Val_Loss_Coord: {:.4f}'.format(self.loss_val_epoch_coord))

        self.init_coord_loss()     

    def save_model(self, epoch):

        filename = os.path.join(self.CHECKPOINT_DIR, 'checkpoint-iter-{:08d}.pth'.format(self.epoch))
        # model_params = model.state_dict()
        if torch.cuda.device_count() > 1:
            model_params = self.model.module.state_dict()  # Saving nn.DataParallel model
        else:
            model_params = self.model.state_dict()

        torch.save(
            {
                'model_state_dict': model_params,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'total_iter_num': self.total_iter_num,
                # 'epoch_loss': epoch_loss,
                # 'config': config_yaml
            }, filename)


    def train(self):
        for epoch in self.iterator:
            self.epoch = epoch + 1
            print("\n ############## epoch: ",epoch)

            for _, batch in enumerate(tqdm(self.train_loader)):  
                #################################### TRAIN CYCLE ###################################  
                self.model.train()    

                sample_batched = self.transfer_to_device(batch)
                rgbs = sample_batched['rgb']
                sim_xyzs = sample_batched['sim_xyz']
                sim_ds = sample_batched['sim_depth']
                syn_ds = sample_batched['syn_depth']
                coords = sample_batched['nocs_map']
                sem_masks = sample_batched['sem_mask']
                ins_masks = sample_batched['ins_mask']
                ins_without_other_masks = sample_batched['ins_w/o_others_mask']
                
                ################################### forward ########################################
                torch.set_grad_enabled(True)
                torch.autograd.set_detect_anomaly(True)
                # outputs_depth, normal_labels, normal_pred, confidence_sim_ds, confidence_initial, \
                #     pred_sem_seg, pred_coord = self.forward(sample_batched, mode='train')
                pred_sem_seg, pred_coord = self.forward(sample_batched, mode='train')

                self.writer.add_scalar('Train/loss/Train_Loss_Total', self.loss_train_iter, self.total_iter_num)
                ################################### backward #######################################                                                                        
                self.optimizer.zero_grad()
                self.loss_train_iter.backward()
                self.optimizer.step()

                ################################### schedule #######################################
                warmup_iters = 500
                warmup_ratio = 1e-06
                if self.total_iter_num < warmup_iters:
                    k = (1 - self.total_iter_num / warmup_iters) * (1 - warmup_ratio)
                    lr_ = self.base_lr * (1 - k)
                else:
                    lr_ = self.base_lr * (1.0 - self.total_iter_num / self.max_iterations) ** 1.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

                ################################## save train metrics ##############################
                # semantic segmentation
                self.save_iter_sem_seg_metrics(sem_masks, pred_sem_seg, mode='train')
                # nocs
                self.save_iter_coord_metrics(mode='train')
                ################################### save learing rate ##############################
                # Log Current Learning Rate
                current_learning_rate = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_learning_rate, self.total_iter_num)

                ################################### save output images #############################       
                pred_sem_seg_argmax = F.softmax(pred_sem_seg, dim=1)  # [1, 150, 512, 512]
                pred_sem_seg_argmax = pred_sem_seg_argmax.argmax(dim=1)  # [1, 512, 512]

                if  self.total_iter_num % 1000 == 0:
                    '''
                    grid_image = create_grid_image(sim_xyzs.detach().cpu(),
                                                   #outputs_depth.detach().cpu(),
                                                   syn_ds.detach().cpu(),
                                                   rgbs.detach().cpu(),
                                                #   normal_pred.detach().cpu(),
                                                #   normal_labels.detach().cpu(),
                                                #   confidence_sim_ds.detach().cpu(),
                                                #   confidence_initial.detach().cpu(),
                                                    sem_masks.detach().cpu(),
                                                    pred_sem_seg_argmax.unsqueeze(1).detach().cpu(),
                                                    coords.detach().cpu(),
                                                    (pred_coord * ins_without_other_masks.squeeze(3).unsqueeze(1)).detach().cpu(),
                                                   max_num_images_to_save=6)
                    
                    self.writer.add_image('Train', grid_image, self.total_iter_num)
                    '''
                    1
                self.total_iter_num += 1
                
            #################################### VALIDATION CYCLE ###################################
            # if self.total_iter_num != 0 and self.total_iter_num % self.val_interation_interval == 0:
            print('\nValidation:')
            print('=' * 10)
            for _, batch in enumerate(tqdm(self.val_loader)):
                self.model.eval()
                sample_batched = self.transfer_to_device(batch)
                rgbs = sample_batched['rgb']
                sim_xyzs = sample_batched['sim_xyz']
                sim_ds = sample_batched['sim_depth']
                syn_ds = sample_batched['syn_depth']
                coords = sample_batched['nocs_map']
                sem_masks = sample_batched['sem_mask']
                ins_masks = sample_batched['ins_mask']
                ins_without_other_masks = sample_batched['ins_w/o_others_mask']

                with torch.no_grad():
                    # outputs_depth, normal_labels, normal_pred, confidence_sim_ds, confidence_initial, \
                    #     pred_sem_seg, pred_coord = self.forward(sample_batched, mode='val')
                    pred_sem_seg, pred_coord = self.forward(sample_batched, mode='val')

                ######################### save nocs train metrics #####################
                self.save_iter_sem_seg_metrics(sem_masks, pred_sem_seg, mode='val')
                self.save_iter_coord_metrics(mode='val')

            self.save_epoch_sem_seg_metrics()
            self.save_epoch_coord_metrics()
            # pred_sem_seg_argmax = F.softmax(pred_sem_seg, dim=1)  # [1, 150, 512, 512]
            # pred_sem_seg_argmax = pred_sem_seg_argmax.argmax(dim=1)  # [1, 512, 512]
            '''
            grid_image = create_grid_image(sim_xyzs.detach().cpu(),
                                            #outputs_depth.detach().cpu(),
                                            syn_ds.detach().cpu(),
                                            rgbs.detach().cpu(),
                                            #normal_pred.detach().cpu(),
                                            #normal_labels.detach().cpu(),
                                            #confidence_sim_ds.detach().cpu(),
                                            #confidence_initial.detach().cpu(),
                                            sem_masks.detach().cpu(),
                                            pred_sem_seg_argmax.unsqueeze(1).detach().cpu(),
                                            coords.detach().cpu(),
                                            (pred_coord * ins_without_other_masks).detach().cpu(),
                                            max_num_images_to_save=6)
        
            # self.writer.add_image('Validation', grid_image, self.total_iter_num)
            self.writer.add_image('Validation', grid_image, self.epoch) 
            '''
            ###################### Save Checkpoints ###############################
            # Save the model checkpoint every N epochs
            self.save_model(epoch)
            
                
        self.writer.close()
        return "Training Finished!"

    '''
    def inference(self):
        #################################### VALIDATION CYCLE ###################################
        # if self.total_iter_num != 0 and self.total_iter_num % self.val_interation_interval == 0:
        print('\nValidation:')
        print('=' * 10)
        for i, batch in enumerate(tqdm(self.val_loader)):
            self.model.eval()
            sample_batched = self.transfer_to_device(batch)
            rgbs = sample_batched['rgb']
            sim_xyzs = sample_batched['sim_xyz']
            sim_ds = sample_batched['sim_depth']
            syn_ds = sample_batched['syn_depth']
            coords = sample_batched['nocs_map']
            sem_masks = sample_batched['sem_mask']
            ins_masks = sample_batched['ins_mask']
            ins_without_other_masks = sample_batched['ins_w/o_others_mask']

            with torch.no_grad():
                # outputs_depth, normal_labels, normal_pred, confidence_sim_ds, confidence_initial, \
                #     pred_sem_seg, pred_coord = self.forward(sample_batched, mode='val')
                start_time = time.time()
                outputs_depth, normal_labels, normal_pred, confidence_sim_ds, confidence_initial = self.forward(sample_batched, mode='val')
                end_time = time.time()
                print((end_time - start_time) / self.batch_size)

            ######################### save depth restoration train metrics #####################
            self.save_iter_depth_restoration_metrics(syn_ds, outputs_depth, ins_masks, mode='val')
            
            if self.val_data_type == 'real':
                pred_xyzs = depth_to_xyz(outputs_depth, self.fx_real_input, self.fy_real_input, scale_h=outputs_depth.shape[2] / 720., scale_w=outputs_depth.shape[3] / 1280.)
                label_xyzs = depth_to_xyz(syn_ds, self.fx_real_input, self.fy_real_input, scale_h=outputs_depth.shape[2] / 720., scale_w=outputs_depth.shape[3] / 1280.)
            else:
                pred_xyzs = depth_to_xyz(outputs_depth, self.fx_sim, self.fy_sim, scale_h=outputs_depth.shape[2] / 360., scale_w=outputs_depth.shape[3] / 640.)
                label_xyzs = depth_to_xyz(syn_ds, self.fx_sim, self.fy_sim, scale_h=outputs_depth.shape[2] / 360., scale_w=outputs_depth.shape[3] / 640.)
    
            rgb = (rgbs[0].permute(1, 2, 0) * 255).trunc()

            mask = ins_masks[0].squeeze()
            mask_save = copy.deepcopy(rgb)
            mask_save[:, :, 0] = mask * 255
            mask_save[:, :, 1] = mask * 255
            mask_save[:, :, 2] = mask * 255
            mask_save = mask_save.cpu().numpy()
            mask_save = cv2.resize(mask_save, (224, 126))
            cv2.imwrite(os.path.join(self.output_path, str(i)+'_ins_mask.png'), mask_save)
                        
            rgb_save = rgb
            rgb_save[:, :, 0] = rgb[:, :, 2]
            rgb_save[:, :, 1] = rgb[:, :, 1]
            rgb_save[:, :, 2] = rgb[:, :, 0]
            rgb_save = rgb_save.cpu().numpy()
            rgb_save = cv2.resize(rgb_save, (224, 126))
            cv2.imwrite(os.path.join(self.output_path, str(i)+'_rgb.png'), rgb_save)
            rgb = rgb.reshape(rgb.shape[0] * rgb.shape[1], rgb.shape[2])
            
            sim_xyz = sim_xyzs[0].permute(1, 2, 0)
            sim_xyz = sim_xyz.reshape(sim_xyz.shape[0] * sim_xyz.shape[1], sim_xyz.shape[2])
            sim_xyz = torch.cat((sim_xyz, rgb), 1)
            sim_xyz = sim_xyz.cpu().numpy()
            
            pred_xyz = pred_xyzs[0].permute(1, 2, 0)
            pred_xyz = pred_xyz.reshape(pred_xyz.shape[0] * pred_xyz.shape[1], pred_xyz.shape[2])
            pred_xyz = torch.cat((pred_xyz, rgb), 1)
            pred_xyz = pred_xyz.cpu().numpy()
    
            label_xyz = label_xyzs[0].permute(1, 2, 0)
            label_xyz = label_xyz.reshape(label_xyz.shape[0] * label_xyz.shape[1], label_xyz.shape[2])
            label_xyz = torch.cat((label_xyz, rgb), 1)
            label_xyz = label_xyz.cpu().numpy() 
            
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
                
            np.savetxt(os.path.join(self.output_path, str(i)+'_input.txt'), sim_xyz)       
            np.savetxt(os.path.join(self.output_path, str(i)+'_pred.txt'), pred_xyz)       
            np.savetxt(os.path.join(self.output_path, str(i)+'_label.txt'), label_xyz)       

            # self.save_iter_sem_seg_metrics(sem_masks, pred_sem_seg, mode='val')
            # self.save_iter_coord_metrics(mode='val')

        self.save_epoch_depth_restoration_metrics(mode='inference')
        # self.save_epoch_sem_seg_metrics()
        # self.save_epoch_coord_metrics()
    '''

    def inference(self):
        #################################### VALIDATION CYCLE ###################################
        # if self.total_iter_num != 0 and self.total_iter_num % self.val_interation_interval == 0:
        print('\nValidation:')
        print('=' * 10)
        self.pose_fitting_result = []
        print('self.val_loader',len(self.val_loader))
        for i, batch in enumerate(tqdm(self.val_loader)):
            self.model.eval()
            sample_batched = self.transfer_to_device(batch)
            with torch.no_grad():
                # outputs_depth, normal_labels, normal_pred, confidence_sim_ds, confidence_initial, \
                #     pred_sem_seg, pred_coord = self.forward(cache, mode='val')
                output_sem_seg, output_coords = self.forward(sample_batched, mode='val')
            
            rgbs = sample_batched['rgb']
            sim_xyzs = sample_batched['sim_xyz']
            sim_ds = sample_batched['sim_depth']
            outputs_depth = sample_batched['output_depth']
            syn_ds = sample_batched['syn_depth']
            coords = sample_batched['nocs_map']
            sem_masks = sample_batched['sem_mask']
            ins_masks = sample_batched['ins_mask']
            ins_without_other_masks = sample_batched['ins_w/o_others_mask']
            print(outputs_depth.shape)
            if self.val_data_type == 'real':
                pred_xyzs = depth_to_xyz(outputs_depth, self.fx_real_input, self.fy_real_input, scale_h=outputs_depth.shape[2] / 720., scale_w=outputs_depth.shape[3] / 1280.)
                label_xyzs = depth_to_xyz(syn_ds, self.fx_real_input, self.fy_real_input, scale_h=outputs_depth.shape[2] / 720., scale_w=outputs_depth.shape[3] / 1280.)
            else:
                pred_xyzs = depth_to_xyz(outputs_depth, self.fx_sim, self.fy_sim, scale_h=outputs_depth.shape[2] / 360., scale_w=outputs_depth.shape[3] / 640.)
                label_xyzs = depth_to_xyz(syn_ds, self.fx_sim, self.fy_sim, scale_h=outputs_depth.shape[2] / 360., scale_w=outputs_depth.shape[3] / 640.)
            '''
            rgb = (rgbs[0].permute(1, 2, 0) * 255).trunc()

            mask = ins_masks[0].squeeze()
            mask_save = copy.deepcopy(rgb)
            mask_save[:, :, 0] = mask * 255
            mask_save[:, :, 1] = mask * 255
            mask_save[:, :, 2] = mask * 255
            mask_save = mask_save.cpu().numpy()
            mask_save = cv2.resize(mask_save, (224, 126))
            cv2.imwrite(os.path.join(self.output_path, str(i)+'_ins_mask.png'), mask_save)
                        
            rgb_save = rgb
            rgb_save[:, :, 0] = rgb[:, :, 2]
            rgb_save[:, :, 1] = rgb[:, :, 1]
            rgb_save[:, :, 2] = rgb[:, :, 0]
            rgb_save = rgb_save.cpu().numpy()
            rgb_save = cv2.resize(rgb_save, (224, 126))
            cv2.imwrite(os.path.join(self.output_path, str(i)+'_rgb.png'), rgb_save)
            rgb = rgb.reshape(rgb.shape[0] * rgb.shape[1], rgb.shape[2])

            sim_xyz = sim_xyzs[0].permute(1, 2, 0)
            sim_xyz = sim_xyz.reshape(sim_xyz.shape[0] * sim_xyz.shape[1], sim_xyz.shape[2])
            #sim_xyz = torch.cat((sim_xyz, rgb), 1)
            sim_xyz = sim_xyz.cpu().numpy()

            pred_xyz = pred_xyzs[0].permute(1, 2, 0)
            pred_xyz = pred_xyz.reshape(pred_xyz.shape[0] * pred_xyz.shape[1], pred_xyz.shape[2])
            #pred_xyz = torch.cat((pred_xyz, rgb), 1)
            pred_xyz = pred_xyz.cpu().numpy()
    
            label_xyz = label_xyzs[0].permute(1, 2, 0)
            label_xyz = label_xyz.reshape(label_xyz.shape[0] * label_xyz.shape[1], label_xyz.shape[2])
            #label_xyz = torch.cat((label_xyz, rgb), 1)
            label_xyz = label_xyz.cpu().numpy() 
            
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
                
            np.savetxt(os.path.join(self.output_path, str(i)+'_input.txt'), sim_xyz)       
            np.savetxt(os.path.join(self.output_path, str(i)+'_pred.txt'), pred_xyz)       
            np.savetxt(os.path.join(self.output_path, str(i)+'_label.txt'), label_xyz)       
            '''
            
            self.inference_iter_coord_metrics(self.transfer_to_cpu(batch), output_sem_seg.detach().cpu(), output_coords.detach().cpu(),mode='val')

        self.inference_epoch_coord_metrics(mode='inference')
        
    def inference_iter_coord_metrics(self,batch,output_sem_masks, output_coords,mode):
        


        #sim_ds = np.array(sim_ds*1000,dtype=int)
        output_ds = np.array(batch['output_depth'])
        syn_ds = np.array(batch['syn_depth'])
        scale = np.array(batch['scale'])
        rgbs = np.array((batch['rgb']*255),dtype=int).transpose(0,2,3,1)
        syn_ds = syn_ds.squeeze(1)
        output_ds = output_ds.squeeze(1)
        sem_masks = np.array(batch['sem_mask']).squeeze(1)
        #sim_xyzs = np.array(batch['sim_xyz']).transpose(0,2,3,1)

        output_sem_masks = F.softmax(output_sem_masks, dim=1)  # [1, 150, 512, 512]
        output_sem_masks = output_sem_masks.argmax(dim=1)  # [1, 512, 512]

        output_sem_masks = np.array(output_sem_masks)

        output_coords = np.array(output_coords)
        output_coords = output_coords.transpose(0,2,3,1)
        coords = np.array(batch['nocs_map'])
        coords = coords.transpose(0,2,3,1)

        #intrinsics = np.array([[156.2085, 0, 111.825], [0, 156.2085, 62.825], [0, 0, 1]])
        intrinsics = np.zeros((3,3))
        if self.val_data_type =='real':
            img_h = 720 
            img_w = 1280  
            # fx = 502.30
            # fy = 502.30
            fx = self.fx_real_input
            fy = self.fy_real_input
        else :
            img_h = 360
            img_w = 640  
            # fx = 502.30
            # fy = 502.30
            fx = self.fx_sim
            fy = self.fy_sim
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
        # get image scale, (x_s, y_s)
        
        hw = (224 / img_w, 126 / img_h)
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
        #print(camera_params)
        camera_params_pred = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'yres': img_h,
            'xres': img_w,
        }
        

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
        
        
        
        for i in range(self.batch_size) :
            #resize_xyzs = cv2.resize(sim_xyzs[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            #np.savetxt(save_path+'/{}_sim_xyzs_pts.txt'.format(i), resize_xyzs.reshape(224*126,3))
            resize_rgbs = cv2.resize(rgbs[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            resize_syn_ds = cv2.resize(syn_ds[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            #print(output_ds.shape)
            resize_output_ds = cv2.resize(output_ds[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            resize_sem_masks = cv2.resize(sem_masks[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            resize_coords = cv2.resize(coords[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            resize_output_sem_masks = cv2.resize(output_sem_masks[i], (224, 126), interpolation=cv2.INTER_NEAREST)
            resize_output_coords = cv2.resize(output_coords[i], (224, 126), interpolation=cv2.INTER_NEAREST)

            pred_class_ids , pred_scores , pred_masks ,pred_coords ,\
                    pred_boxes = prepare_data_posefitting(resize_output_sem_masks,resize_output_coords)

            gt_class_ids , gt_scores , gt_masks ,gt_coords ,\
                    gt_boxes = prepare_data_posefitting(resize_sem_masks,resize_coords)
            if len(pred_class_ids) == 0 or len(gt_class_ids) == 0 :
                continue
            result = {}


            result['gt_class_ids'] = gt_class_ids
            result['gt_bboxes'] = gt_boxes

            result['gt_RTs'] = None  
      
            result['gt_scales'] = scale[i][gt_class_ids]

            result['gt_RTs'], scales, error_message, _ = align(gt_class_ids, 
                                                        gt_masks, 
                                                        gt_coords, 
                                                        resize_syn_ds, 
                                                        intrinsics, 
                                                        synset_names,save_path='tmp/gts/')

            result['pred_class_ids'] = pred_class_ids
            result['pred_bboxes'] = pred_boxes
            result['pred_RTs'] = None   
            result['pred_scores'] = pred_scores
            result['pred_RTs'], result['pred_scales'], error_message, elapses =  align(pred_class_ids, 
                                                                                        pred_masks, 
                                                                                        pred_coords, 
                                                                                        resize_output_ds, 
                                                                                        intrinsics, 
                                                                                        synset_names, 
                                                                                        if_norm=True,save_path='tmp/pred/')
            

            
            
            self.pose_fitting_result.append(result)
            
            if 1:
                draw_rgb = False
                save_dir =os.path.join(self.output_path ,'save_{}'.format(i))
                if not os.path.exists(save_dir) :
                    os.mkdir(save_dir)
                data = 'camera'
                result['gt_handle_visibility'] = np.ones_like(gt_class_ids)

                draw_detections(resize_rgbs, save_dir, data, 1, intrinsics, synset_names, draw_rgb,
                                        gt_boxes, gt_class_ids, gt_masks, gt_coords, result['gt_RTs'], result['gt_scales'], result['gt_handle_visibility'],
                                        pred_boxes, pred_class_ids, pred_masks, pred_coords, result['pred_RTs'], pred_scores, result['pred_scales'])
            exit(0)
    
    def inference_epoch_coord_metrics(self,mode):
        synset_names = ['other',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'car',  # 5
                    'mug',  # 6
                    'aeroplane',  # 7
                    #'BG',  # 8
                    ]
        save_path =os.path.join(self.output_path ,'result.pkl') 
        with open(save_path, 'wb') as f:
            cPickle.dump(self.pose_fitting_result, f)
        print(save_path)
        aps = compute_degree_cm_mAP(self.pose_fitting_result, synset_names, self.output_path ,
                                                                degree_thresholds = [5, 10, 15], 
                                                                shift_thresholds= [2,5, 10, 15],  
                                                                iou_3d_thresholds=np.linspace(0, 1, 101),
                                                                iou_pose_thres=0.1,
                                                                use_matches_for_pose=True)
  
