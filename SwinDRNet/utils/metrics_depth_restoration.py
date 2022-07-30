import numpy as np
import torch
import cv2
import torch.nn.functional as F

def get_metrics_depth_restoration_train(gt, pred, width, height, seg_mask=None):

    gt = gt.detach().permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    pred = pred.detach().permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    if not seg_mask is None:
        seg_mask = seg_mask.detach().cpu().permute(0, 2, 3, 1).numpy()

    gt_depth = gt
    pred_depth = pred
    gt_depth[np.isnan(gt_depth)] = 0
    gt_depth[np.isinf(gt_depth)] = 0
    mask_valid_region = (gt_depth > 0)
    
    if not seg_mask is None:
        seg_mask = seg_mask.astype(np.uint8)
        mask_valid_region = np.logical_and(mask_valid_region, seg_mask)

    gt = torch.from_numpy(gt_depth).float().cuda()
    pred = torch.from_numpy(pred_depth).float().cuda()
    mask = torch.from_numpy(mask_valid_region).bool().cuda()
    gt = gt[mask]
    pred = pred[mask]

    thresh = torch.max(gt / pred, pred / gt)
    
    a1 = (thresh < 1.05).float().mean()
    a2 = (thresh < 1.10).float().mean()
    a3 = (thresh < 1.25).float().mean()
    
    rmse = ((gt - pred)**2).mean().sqrt()
    abs_rel = ((gt - pred).abs() / gt).mean()
    mae = (gt - pred).abs().mean()

    return a1, a2, a3, rmse, abs_rel, mae


def get_metrics_depth_restoration_inference(gt, pred, width, height, seg_mask=None):
    B = gt.shape[0]
    gt = F.interpolate(gt, size=[width, height], mode="nearest")
    pred = F.interpolate(pred, size=[width, height], mode="nearest")

    gt = gt.detach().permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    pred = pred.detach().permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    if not seg_mask is None:
        seg_mask = seg_mask.float()
        seg_mask = F.interpolate(seg_mask, size=[width, height], mode="nearest")
        seg_mask = seg_mask.detach().cpu().permute(0, 2, 3, 1).numpy()

    gt_depth = gt
    pred_depth = pred
    gt_depth[np.isnan(gt_depth)] = 0
    gt_depth[np.isinf(gt_depth)] = 0
    mask_valid_region = (gt_depth > 0)
    
    if not seg_mask is None:
        seg_mask = seg_mask.astype(np.uint8)
        mask_valid_region = np.logical_and(mask_valid_region, seg_mask)

    gt = torch.from_numpy(gt_depth).float().cuda()
    pred = torch.from_numpy(pred_depth).float().cuda()
    mask = torch.from_numpy(mask_valid_region).bool().cuda()
    
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    rmse = 0.0
    abs_rel = 0.0
    mae = 0.0    
    
    num_valid = 0
    
    for i in range(B):
        gt_i = gt[i][mask[i]]
        pred_i = pred[i][mask[i]]
        # print(len(gt_i))

        if len(gt_i) > 0:
            num_valid += 1
            thresh = torch.max(gt_i / pred_i, pred_i / gt_i)
            
            a1_i = (thresh < 1.05).float().mean()
            a2_i = (thresh < 1.10).float().mean()
            a3_i = (thresh < 1.25).float().mean()
            
            rmse_i = ((gt_i - pred_i)**2).mean().sqrt()
            abs_rel_i = ((gt_i - pred_i).abs() / gt_i).mean()
            mae_i = (gt_i - pred_i).abs().mean()
            a1 += a1_i
            a2 += a2_i
            a3 += a3_i
            rmse += rmse_i
            abs_rel += abs_rel_i
            mae += mae_i   
            # print(a1.item(), a2.item(), a3.item(), rmse.item(), abs_rel.item(), mae.item())
        
    return a1, a2, a3, rmse, abs_rel, mae, num_valid