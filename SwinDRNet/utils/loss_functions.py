'''
This module contains the loss functions used to train the surface normals estimation models.
'''

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from termcolor import colored
from torch.nn.modules import loss


def loss_fn_l1_with_mask(input_vec, target_vec, masks=None, reduction='sum'):
    B, _, H, W = target_vec.shape
    l1 = nn.L1Loss(reduction='none')
    loss_l1 = l1(input_vec, target_vec)
    loss_l1 = torch.sum(loss_l1, dim=1, keepdim=True)
    if masks == None:
        return torch.sum(loss_l1) / (B * H * W), 0
    else:
        num_instance = torch.sum(masks)
        num_background = torch.sum(1 - masks)
        loss_instance = torch.sum(loss_l1 * masks) / num_instance
        loss_background = torch.sum(loss_l1 * (1 - masks)) / num_background
        return loss_instance, loss_background


def loss_fn_l1(input_vec, target_vec, reduction='sum'):
    l1 = nn.L1Loss(reduction='none')
    loss_l1 = l1(input_vec, target_vec)
    loss_l1 = torch.sum(loss_l1, dim=1, keepdim=False)
    '''
    # calculate loss only on valid pixels
    # mask_invalid_pixels = (target_vec[:, 0, :, :] == -1.0) & (target_vec[:, 1, :, :] == -1.0) & (target_vec[:, 2, :, :] == -1.0)
    mask_invalid_pixels = torch.all(target_vec == -1, dim = 1) & torch.all(target_vec == 0, dim = 1)

    loss_l1[mask_invalid_pixels] = 0.0
    loss_l1_sum = loss_l1.sum()
    total_valid_pixels = (~mask_invalid_pixels).sum()
    error_output = loss_l1_sum / total_valid_pixels

    if reduction == 'elementwise_mean':
        loss_l1 = error_output
    elif reduction == 'sum':
        loss_l1 = loss_l1_sum
    elif reduction == 'none':
        loss_l1 = loss_l1
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())
    '''
    return loss_l1.sum()


def loss_fn_cosine(input_vec, target_vec, reduction='sum'):
    '''A cosine loss function for use with surface normals estimation.
    Calculates the cosine loss between 2 vectors. Both should be of the same size.

    Arguments:
        input_vec {tensor} -- The 1st vectors with whom cosine loss is to be calculated
                              The dimensions of the matrices are expected to be (batchSize, 3, height, width).
        target_vec {tensor } -- The 2nd vectors with whom cosine loss is to be calculated
                                The dimensions of the matrices are expected to be (batchSize, 3, height, width).

    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})

    Raises:
        Exception -- Exception is an invalid reduction is passed

    Returns:
        tensor -- A single mean value of cosine loss or a matrix of elementwise cosine loss.
    '''
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = 1.0 - cos(input_vec, target_vec)

    # calculate loss only on valid pixels
    # mask_invalid_pixels = (target_vec[:, 0, :, :] == -1.0) & (target_vec[:, 1, :, :] == -1.0) & (target_vec[:, 2, :, :] == -1.0)
    mask_invalid_pixels = torch.all(target_vec == -1, dim=1) & torch.all(target_vec == 0, dim=1)

    loss_cos[mask_invalid_pixels] = 0.0
    loss_cos_sum = loss_cos.sum()
    total_valid_pixels = (~mask_invalid_pixels).sum()
    error_output = loss_cos_sum / total_valid_pixels

    if reduction == 'elementwise_mean':
        loss_cos = error_output
    elif reduction == 'sum':
        loss_cos = loss_cos_sum
    elif reduction == 'none':
        loss_cos = loss_cos
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_cos


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        edge_ky = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        edge_k = np.stack((edge_kx, edge_ky))

        #edge_k = torch.from_numpy(edge_k).double().view(2, 1, 3, 3)
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        n, c, h, w = x.shape
            x is n examples, each have h*w pixels, and each pixel contain c=1 channel value

        n, 2, h, w = out.shape
            2 channel: first represents dx, second represents dy
        """
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel().cuda()

    def forward(self, input, target):
        input_grad = self.sobel(input)
        target_grad = self.sobel(target)
        #  n, 2, h, w = out.shape
        #  2 channel: first represents dx, second represents dy

        input_grad_dx = input_grad[:, 0, :, :].contiguous().view_as(input)
        input_grad_dy = input_grad[:, 1, :, :].contiguous().view_as(input)
        target_grad_dx = target_grad[:, 0, :, :].contiguous().view_as(target)
        target_grad_dy = target_grad[:, 1, :, :].contiguous().view_as(target)

        # loss_dx = torch.abs(input_grad_dx - target_grad_dx).mean()#torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5).mean()
        # loss_dy = torch.abs(input_grad_dy - target_grad_dy).mean()#torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5).mean()


        loss_dx = torch.abs(input_grad_dx - target_grad_dx)#torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5).mean()
        loss_dy = torch.abs(input_grad_dy - target_grad_dy)#torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5).mean()


        # loss_dx = torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5)
        # loss_dy = torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5)     
         
        # loss_dx = torch.log(torch.abs(input_grad_dx - target_grad_dx) + 1.0)
        # loss_dy = torch.log(torch.abs(input_grad_dy - target_grad_dy) + 1.0)

        return loss_dx + loss_dy





def metric_calculator_batch(input_vec, target_vec, mask=None):
    """Calculate mean, median and angle error between prediction and ground truth

    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (batchSize, 3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (batchSize, 3, height, width).
        mask (tensor): The pixels over which loss is to be calculated. Represents VALID pixels.
                             The dimensions are expected to be (batchSize, 3, height, width).

    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 3 degrees

    """
    if len(input_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(target_vec.shape))

    INVALID_PIXEL_VALUE = 0  # All 3 channels should have this value
    mask_valid_pixels = ~(torch.all(target_vec == INVALID_PIXEL_VALUE, dim=1))
    if mask is not None:
        mask_valid_pixels = (mask_valid_pixels.float() * mask).byte()
    total_valid_pixels = mask_valid_pixels.sum()
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)

    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    loss_deg = loss_deg[mask_valid_pixels]
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg < 30).sum().float() / total_valid_pixels) * 100

    return loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3


def metric_calculator(input_vec, target_vec, mask=None):
    """Calculate mean, median and angle error between prediction and ground truth

    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (3, height, width).
        mask (tensor): Optional mask of area where loss is to be calculated. All other pixels are ignored.
                       Shape: (height, width), dtype=uint8

    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 3 degrees

    """
    if len(input_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(target_vec.shape))

    INVALID_PIXEL_VALUE = 0  # All 3 channels should have this value
    mask_valid_pixels = ~(torch.all(target_vec == INVALID_PIXEL_VALUE, dim=0))
    if mask is not None:
        mask_valid_pixels = (mask_valid_pixels.float() * mask).byte()
    total_valid_pixels = mask_valid_pixels.sum()
    # TODO: How to deal with a case with zero valid pixels?
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)

    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    loss_deg = loss_deg[mask_valid_pixels]
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg < 30).sum().float() / total_valid_pixels) * 100

    return loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels


# TODO: Fix the loss func to ignore invalid pixels
def loss_fn_radians(input_vec, target_vec, reduction='sum'):
    '''Loss func for estimation of surface normals. Calculated the angle between 2 vectors
    by taking the inverse cos of cosine loss.

    Arguments:
        input_vec {tensor} -- First vector with whole loss is to be calculated.
                              Expected size (batchSize, 3, height, width)
        target_vec {tensor} -- Second vector with whom the loss is to be calculated.
                               Expected size (batchSize, 3, height, width)

    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})

    Raises:
        Exception -- If any unknown value passed as reduction argument.

    Returns:
        tensor -- Loss from 2 input vectors. Size depends on value of reduction arg.
    '''

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)
    loss_rad = torch.acos(loss_cos)
    if reduction == 'elementwise_mean':
        loss_rad = torch.mean(loss_rad)
    elif reduction == 'sum':
        loss_rad = torch.sum(loss_rad)
    elif reduction == 'none':
        pass
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_rad


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    """
    The loss is

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

        `(minibatch, C, d_1, d_2, ..., d_K)`

    Args:
        logit (Tensor): Output of network
        target (Tensor): Ground Truth
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss
        weight (List, optional): Defaults to None. Weight assigned to each class
        batch_average (bool, optional): Defaults to True. Whether to consider the loss of each element in the batch.

    Returns:
        Float: The value of loss.
    """

    n, c, h, w = logit.shape
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32),
                                        ignore_index=ignore_index,
                                        reduction='sum')

    loss = criterion(logit, target.long())

    if batch_average:
        loss /= n

    return loss


def metric_calculator_batch_with_mask(input_vec, target_vec, mask=None):
    """Calculate mean, median and angle error between prediction and ground truth

    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (batchSize, 3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (batchSize, 3, height, width).
        mask (tensor): The pixels over which loss is to be calculated. Represents VALID pixels.
                             The dimensions are expected to be (batchSize, 3, height, width).

    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 3 degrees

    """
    if len(input_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(target_vec.shape))

    INVALID_PIXEL_VALUE = 0  # All 3 channels should have this value
    mask_valid_pixels = ~(torch.all(target_vec == INVALID_PIXEL_VALUE, dim=1)).cuda()


    if mask is not None:
        mask_valid_pixels_instance = (mask_valid_pixels.float() * mask).byte()
        mask_valid_pixels_background = (mask_valid_pixels.float() * (1 - mask)).byte()


    total_valid_pixels = mask_valid_pixels_instance.sum()
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)

    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    loss_deg_instance = loss_deg[mask_valid_pixels_instance]
    loss_deg_mean = loss_deg_instance.mean()
    loss_deg_median = loss_deg_instance.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg_instance < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg_instance < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg_instance < 30).sum().float() / total_valid_pixels) * 100

    loss_instance = (loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3)


    total_valid_pixels = mask_valid_pixels_background.sum()
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    # Mask out all invalid pixels and calc mean, median
    loss_deg_background = loss_deg[mask_valid_pixels_background]
    loss_deg_mean = loss_deg_background.mean()
    loss_deg_median = loss_deg_background.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg_background < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg_background < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg_background < 30).sum().float() / total_valid_pixels) * 100

    loss_background = (loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3)

  
    return loss_instance, loss_background


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
