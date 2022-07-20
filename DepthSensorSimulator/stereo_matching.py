import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import kornia
import copy
from torch.utils.data import Dataset
import glob
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def GetGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (torch.arange(ksize, dtype=torch.double, device=sigma.device) - center)
    kernel1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = kernel / kernel.sum()
    return kernel


def BilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    
    # calculate normalized weight matrix
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # obtain the gaussian kernel
    weights_space = GetGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # caluculate the final weight
    weights = weights_space * weights_color
    weights_sum = weights.sum(dim=(-1, -2))
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


# Left corr without zero mean
def CorrLWithoutZeroMean(i, cacheImageL, cacheImageR, filters, padding, eps):
    imageL, _, _, _, _, imageL2Sum = cacheImageL
    imageR, _, _, _, _, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedR = imageR.narrow(3, 0, W - i)                
    cropedR2Sum = imageR2Sum.narrow(3, 0, W - i)        

    shifted = F.pad(cropedR, (i, 0, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedR2Sum, (i, 0, 0, 0), "replicate")

    product = shifted * imageL
    productSum = F.conv2d(product, filters, stride=1, padding=padding)
    corrL = (productSum + eps) / (imageL2Sum.sqrt() * shifted2Sum.sqrt() + eps)
    return corrL


# Right corr without zero mean
def CorrRWithoutZeroMean(i, cacheImageL, cacheImageR, filters, padding, eps):
    imageL, _, _, _, _, imageL2Sum = cacheImageL
    imageR, _, _, _, _, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedL = imageL.narrow(3, i, W - i)  
    cropedL2Sum = imageL2Sum.narrow(3, i, W - i)  
    shifted = F.pad(cropedL, (0, i, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedL2Sum, (0, i, 0, 0), "replicate")
    product = shifted * imageR
    productSum = F.conv2d(product, filters, stride=1, padding=padding).double()
    corrR = (productSum + eps) / (imageR2Sum.sqrt() * shifted2Sum.sqrt() + eps)

    return corrR


# Left Corr
def CorrL(i, cacheImageL, cacheImageR, filters, padding, blockSize, eps):
    imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum = cacheImageL
    imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedR = imageR.narrow(3, 0, W - i)                
    cropedRSum = imageRSum.narrow(3, 0, W - i)          
    cropedR2Sum = imageR2Sum.narrow(3, 0, W - i)        
    cropedRAve = imageRAve.narrow(3, 0, W - i)          
    cropedRAve2 = imageRAve2.narrow(3, 0, W - i)        

    shifted = F.pad(cropedR, (i, 0, 0, 0), "constant", 0.0)
    shiftedSum = F.pad(cropedRSum, (i, 0, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedR2Sum, (i, 0, 0, 0), "constant", 0.0)
    shiftedAve = F.pad(cropedRAve, (i, 0, 0, 0), "constant", 0.0)
    shiftedAve2 = F.pad(cropedRAve2, (i, 0, 0, 0), "constant", 0.0)

    LShifted = imageL * shifted
    LShiftedSum = F.conv2d(LShifted, filters, stride=1, padding=padding).double()
    LAveShifted = imageLAve * shiftedSum
    shiftedAveL = shiftedAve * imageLSum
    LAveShiftedAve = imageLAve * shiftedAve
    productSum = LShiftedSum - LAveShifted - shiftedAveL + blockSize * blockSize * C * LAveShiftedAve

    sqrtL = (imageL2Sum - 2 * imageLAve * imageLSum + blockSize * blockSize * C * imageLAve2 + 1e-5).sqrt()
    sqrtShifted = (shifted2Sum - 2 * shiftedAve * shiftedSum + blockSize * blockSize * C * shiftedAve2 + 1e-5).sqrt()
    
    corrL = (productSum + eps) / (sqrtL * sqrtShifted + eps)
    corrL[:, :, :, :i] = 0

    return corrL

# Right Corr
def CorrR(i, cacheImageL, cacheImageR, filters, padding, blockSize, eps):
    imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum = cacheImageL
    imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedL = imageL.narrow(3, i, W - i)                
    cropedLSum = imageLSum.narrow(3, i, W - i)          
    cropedL2Sum = imageL2Sum.narrow(3, i, W - i)       
    cropedLAve = imageLAve.narrow(3, i, W - i)         
    cropedLAve2 = imageLAve2.narrow(3, i, W - i)       

    shifted = F.pad(cropedL, (0, i, 0, 0), "constant", 0.0)
    shiftedSum = F.pad(cropedLSum, (0, i, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedL2Sum, (0, i, 0, 0), "constant", 0.0)
    shiftedAve = F.pad(cropedLAve, (0, i, 0, 0), "constant", 0.0)
    shiftedAve2 = F.pad(cropedLAve2, (0, i, 0, 0), "constant", 0.0)

    RShifted = imageR * shifted
    RShiftedSum = F.conv2d(RShifted, filters, stride=1, padding=padding).double()
    RAveShifted = imageRAve * shiftedSum
    shiftedAveR = shiftedAve * imageRSum
    RAveShiftedAve = imageRAve * shiftedAve
    productSum = RShiftedSum - RAveShifted - shiftedAveR + blockSize * blockSize * C * RAveShiftedAve

    sqrtR = (imageR2Sum - 2 * imageRAve * imageRSum + blockSize * blockSize * C * imageRAve2 + 1e-5).sqrt()
    sqrtShifted = (shifted2Sum - 2 * shiftedAve * shiftedSum + blockSize * blockSize * C * shiftedAve2 + 1e-5).sqrt()
    
    corrR = (productSum + eps) / (sqrtR * sqrtShifted + eps)
    corrR[:, :, :, W - i:] = 0

    return corrR


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, dtype=x.dtype, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, dtype=x.dtype, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)

    vgrid = grid + flo  # B,2,H,W

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones(x.size(), dtype=x.dtype, device=x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


def LRC(dispL, dispR):
    dispRClone = dispR.clone()
    dispSamplesX = dispL.permute(0, 2, 3, 1)
    dispSamplesY = torch.zeros_like(dispSamplesX)
    dispSamples = torch.cat((-dispSamplesX, dispSamplesY), dim=-1)
    dispSamples = dispSamples.permute(0, 3, 1, 2)

    wrapedDispR = warp(dispRClone, dispSamples)
    disp = dispL.clone()
    disp[torch.pow((dispL - wrapedDispR), 2) > 0.5] = -1.0
    return disp


def LRC1(dispL, dispR):
    device = dispL.device
    B, C, H, W = dispL.shape    # C = 1
    dispSamplesX = dispL.permute(0, 2, 3, 1)

    indexX = torch.arange(0, W, 1, device=device)
    indexX = indexX.repeat(H, 1)
    indexY = torch.arange(0, H, 1, device=device)
    indexY = indexY.repeat(W, 1).transpose(0, 1)

    indexX = indexX.repeat(B, 1, 1, 1).permute(0, 2, 3, 1)
    indexY = indexY.repeat(B, 1, 1, 1).permute(0, 2, 3, 1)

    dispSamplesX = indexX - dispSamplesX

    dispSamplesY = indexY
    dispSamples = torch.cat((dispSamplesX, dispSamplesY), dim=-1)

    dispSamples[:, :, :, 0] = 2.0 * dispSamples[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
    dispSamples[:, :, :, 1] = 2.0 * dispSamples[:, :, :, 1].clone() / max(H - 1, 1) - 1.0

    wrapedDispR = torch.nn.functional.grid_sample(dispR, dispSamples, align_corners=True)
    disp = dispL.clone()
    disp[torch.pow((dispL - wrapedDispR), 2) > 1] = 0

    return disp


# costVolume to disparity
# costVolume [D B C H W], dispVolume [D B C H W]
def CostToDisp(costVolume, dispVolume, beta, eps, subPixel):
    # sub = d + (c1 - c2)/(2*(c1 + c2 - 2*c0))
    if subPixel == True:
        D, B, C, H, W = costVolume.shape
        costVolumePad = torch.full((1, B, 1, H, W), 0, device=costVolume.device)  # padding

        dispVolume = (dispVolume + (torch.cat((costVolumePad, costVolume.narrow(0, 0, D - 1)))
                                   - torch.cat((costVolume.narrow(0, 1, D - 1), costVolumePad)) + eps) / \
                                  (2 * (torch.cat((costVolumePad, costVolume.narrow(0, 0, D - 1)))
                                   + torch.cat((costVolume.narrow(0, 1, D - 1), costVolumePad)) - 2 * costVolume) + eps))

    softmaxAttention = F.softmax(costVolume * beta, dim=0)                  # [D B C H W]
    dispVolume = (softmaxAttention * dispVolume).permute(1, 2, 3, 4, 0)     # [B C H W D]

    return torch.sum(dispVolume, 4)


def DispToDepth(dispImage, f, baselineDis, eps):
    depthImage = f * baselineDis / (dispImage + eps)
    return depthImage


def DepthToPointCloud(depthImage, f):
    B, C, H, W = depthImage.shape
    device = depthImage.device
    du = W//2 - 0.5
    dv = H//2 - 0.5

    pointCloud = torch.zeros([B, H, W, 3], device=device)
    imageIndexX = -(torch.arange(0, W, 1, device=device) - du)
    imageIndexY = -(torch.arange(0, H, 1, device=device) - dv)
    depthImage = depthImage.squeeze()
    if B == 1:
        depthImage = depthImage.unsqueeze(0)

    pointCloud[:, :, :, 0] = depthImage/f * imageIndexX
    pointCloud[:, :, :, 1] = (depthImage.transpose(1, 2)/f * imageIndexY.T).transpose(1, 2)
    pointCloud[:, :, :, 2] = depthImage
    pointCloud = pointCloud.view(B, H*W, 3)
    return pointCloud


def lerp(a,b,x):
    return a + x * (b-a)


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h,x,y):
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]], dtype=np.float32)
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y


def perlin(x, y, seed=0):
    # permutation table
    if seed != None:
        np.random.seed(seed)
    p = np.arange(640, dtype=np.int32)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(np.int32)
    yi = y.astype(np.int32)
    # internal coordinates
    xf = (x - xi).astype(np.float32)
    yf = (y - yi).astype(np.float32)
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi], xf, yf)
    n01 = gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = gradient(p[p[xi+1]+yi], xf-1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)


def post_processing(disparity_map, diff_insame, min_speckle_aera):
    B, C, H, W = disparity_map.shape

    for b in range(B):
        mask_visited = torch.zeros((H, W), device=disparity_map.device)
        for i in range(H):
            for j in range(W):
                print(b, i, j)
                if mask_visited[i, j] == 1 or disparity_map[b, :, i, j] < 0:
                    continue
                vec = []
                vec.append([i, j])
                mask_visited[i, j] = 1
                cur = 0
                next = 0
                while True:
                    next = len(vec)
                    print(len(vec))
                    for k in range(cur, next, 1):
                        pixel = vec[k]
                        row = pixel[0]
                        col = pixel[1]
                        disp_base = disparity_map[b, :, row, col]
                        for r in range(-1, 2, 1):
                            for c in range(-1, 2, 1):
                                if r == 0 and c == 0:
                                    continue
                                rowr = row + r
                                colc = col + c
                                if rowr >= 0 and rowr < H and colc >= 0 and colc < W:
                                    if mask_visited[rowr, colc] == 0 and torch.abs(disparity_map[b, :, rowr, colc] - disp_base) <= diff_insame:
                                        vec.append([rowr, colc])
                                        mask_visited[rowr, colc] = 1
                    cur = next
                    if next >= len(vec):
                        break
                if len(vec) < min_speckle_aera:
                    for v in vec:
                        disparity_map[b, :, v[0], v[1]] = -1.0
    return disparity_map


class StereoMatching(nn.Module):
    def __init__(self, maxDisp = 60, minDisp = 1, blockSize = 9, eps = 1e-6, subPixel = True, bilateralFilter = True):
        super(StereoMatching, self).__init__()
        self.maxDisp = maxDisp
        self.minDisp = minDisp
        self.blockSize = blockSize
        self.eps = eps
        self.subPixel = subPixel
        self.bilateralFilter = bilateralFilter

    def forward(self, imageL, imageR, f, blDis, beta, sigmaColor=0.05, sigmaSpace=5):
        print("forward start")
        beginTime = time.time()
        # imageL = imageL.type(torch.FloatTensor).cuda()
        # imageR = imageR.type(torch.FloatTensor).cuda()
        B, C, H, W = imageL.shape
        D = self.maxDisp - self.minDisp + 1
        device = imageL.device

        if(self.maxDisp >= imageR.shape[3]):
            raise RuntimeError("The max disparity must be smaller than the width of input image!")

        # Normal distribution noise
        mu = 0.0
        sigma = 1.0
        mu_t = torch.full(imageL.size(), mu)
        std = torch.full(imageL.size(), sigma) 
        eps_l = torch.randn_like(std)  
        eps_r = torch.randn_like(std)  

        if imageL.is_cuda:
            mu_t = mu_t.cuda()
            std = std.cuda()
            eps_l = eps_l.cuda()
            eps_r = eps_r.cuda()

        delta_img_receiver_l = eps_l.mul(std).add_(mu_t)
        delta_img_receiver_r = eps_r.mul(std).add_(mu_t)
        delta_img_receiver_l[delta_img_receiver_l < 0] = 0
        delta_img_receiver_r[delta_img_receiver_r > 255] = 255

        dispVolume = torch.zeros([D, B, 1, H, W], device=device)         # [B C H W D]
        costVolumeL = torch.zeros([D, B, 1, H, W], device=device)        # [B C H W D]
        costVolumeR = torch.zeros([D, B, 1, H, W], device=device)        # [B C H W D]


        filters = Variable(torch.ones(1, C, self.blockSize, self.blockSize, dtype=imageL.dtype, device=device))
        padding = (self.blockSize // 2, self.blockSize // 2)

        imageLSum = F.conv2d(imageL, filters, stride=1, padding=padding)
        imageLAve = imageLSum/(self.blockSize * self.blockSize * C)
        imageLAve2 = imageLAve.pow(2)
        imageL2 = imageL.pow(2)
        imageL2Sum = F.conv2d(imageL2, filters, stride=1, padding=padding)

        imageRSum = F.conv2d(imageR, filters, stride=1, padding=padding)
        imageRAve = imageRSum/(self.blockSize * self.blockSize * C)
        imageRAve2 = imageRAve.pow(2)
        imageR2 = imageR.pow(2)
        imageR2Sum = F.conv2d(imageR2, filters, stride=1, padding=padding)

        cacheImageL = [imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum]
        cacheImageR = [imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum]

        # calculate costVolume
        testBeginTime = time.time()
        for i in range(self.minDisp, D + 1, 1):
            # ConcostVolume and dispVolume
            costVolumeL[i - self.minDisp] = CorrL(i, cacheImageL, cacheImageR, filters, padding, self.blockSize, self.eps)
            costVolumeR[i - self.minDisp] = CorrR(i, cacheImageL, cacheImageR, filters, padding, self.blockSize, self.eps)
            dispVolume[i - self.minDisp] = torch.full_like(costVolumeL[0], i)

        testEndTime = time.time()
        print("costVolume Time: ", testEndTime - testBeginTime)

        # calculate disparity map
        testBeginTime = time.time()
        dispL = CostToDisp(costVolumeL, dispVolume, beta, self.eps, self.subPixel)
        dispR = CostToDisp(costVolumeR, dispVolume, beta, self.eps, self.subPixel)

        testEndTime = time.time()
        print("costToDisp Time: ", testEndTime - testBeginTime)

        # dispL[dispL < self.minDisp] = -1.0
        # dispL[dispL > self.maxDisp] = -1.0
        # dispR[dispR < self.minDisp] = -1.0
        # dispR[dispR > self.maxDisp] = -1.0

        dispLRC = LRC(dispL, dispR)
        disp = dispLRC

        if self.bilateralFilter == True:
            # disp: torch.Tensor = kornia.median_blur(disp, (5, 5))
            disp = kornia.filters.median_blur(disp, (5, 5))
            disp = BilateralFilter(disp, 7, sigmaColor=sigmaColor * D, sigmaSpace=sigmaSpace)

        disp[disp < self.minDisp] = -1.0
        disp[disp > self.maxDisp] = -1.0
        # disp = post_processing(disp, 2, 100)
        # disparity to depth
        depthImage = DispToDepth(disp, f, blDis, self.eps)  # [B H W]

        # depthImage[depthImage > 4000] = 0.0
        depthImage[depthImage < 0] = -0.001
        depthImage[depthImage > 3.5] = -0.001

        # depth to point cloud
        pointCloud = DepthToPointCloud(depthImage, f)
        endTime = time.time()
        print("forward finish")
        print("forward time: ", endTime - beginTime)
        return depthImage, pointCloud


class DREDS(Dataset):
    def __init__(self, root, scale):
        self.root = root
        self.scale = scale
        self.ir_l_list = []
        self.ir_r_list = []
        self.depth_list = []
        for img_id in glob.glob(self.root+"/*/*_ir_l.png"):
            if not os.path.exists(img_id[:-8] + "simDepthImage.exr"):
                path = img_id[:-8]
                self.ir_l_list.append(path + "ir_l.png")
                self.ir_r_list.append(path + "ir_r.png")
                self.depth_list.append(path + "depth_120.exr")
        print(self.ir_l_list, self.ir_r_list, self.depth_list)

    def __getitem__(self, index):
        # read ir image to tensor
        ir_l_path = self.ir_l_list[index]
        ir_r_path = self.ir_r_list[index]
        imageL = Image.open(ir_l_path).convert("L")
        imageR = Image.open(ir_r_path).convert("L")
        imageL = np.array(imageL.resize((int(imageL.size[0] * self.scale), int(imageL.size[1] * self.scale)))).astype(np.float32)
        imageR = np.array(imageR.resize((int(imageR.size[0] * self.scale), int(imageR.size[1] * self.scale)))).astype(np.float32)
        imageL = np.array([imageL]).transpose(1, 2, 0)
        imageR = np.array([imageR]).transpose(1, 2, 0)
        imageLTensor = torch.from_numpy(imageL.transpose(2, 0, 1))
        imageRTensor = torch.from_numpy(imageR.transpose(2, 0, 1))

        # read depth to tensor
        depth_image_path = self.depth_list[index]
        image = cv2.imread(depth_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_array = image[:, :, 0]
        depthTensor = torch.from_numpy(depth_array).unsqueeze(0)
        return imageLTensor, imageRTensor, ir_l_path, ir_r_path, depthTensor

    def __len__(self):
        return len(self.ir_l_list)


def main_batch():
    ###########################
    # Parameter setting
    ###########################
    scale = 0.5                             # image scale
    beta = torch.tensor(100.)               # beta para of softArgmax
    blockSize =11                           # stereo matching para
    maxDisp = 110                           # max disparity between left and right images
    minDisp = 3                             # min disparity between left and right images
    f = torch.tensor(446.31)                # focal length of the sensor
    baselineDis = torch.tensor(0.055)       # baseline distance of the sensor
    subPixel = True                         # whether to calculate subpixel
    bilateralFilter = True                  # whether to use bilateral filter
    sigmaColor = torch.tensor(0.02)         # color para of bilateral filter
    sigmaSpace = torch.tensor(3.)           # space para of bilateral filter
    gpu_id = 6
    batch_size = 8
    input_root = "./rendered_output"
    save_root = "./rendered_output"

    ###########################
    # Main
    ###########################
    torch.cuda.set_device(gpu_id)

    DREDS_DATASET = DREDS(root=input_root, scale=scale)
    print("Len: ", len(DREDS_DATASET))
    DataLoader = torch.utils.data.DataLoader(DREDS_DATASET, batch_size=batch_size, shuffle=False, num_workers=10)

    beta = torch.autograd.Variable(beta)
    sigmaColor = torch.autograd.Variable(sigmaColor)
    sigmaSpace = torch.autograd.Variable(sigmaSpace)

    object = StereoMatching(maxDisp=maxDisp, minDisp=minDisp, blockSize=blockSize, eps=1e-6, subPixel=subPixel,
                            bilateralFilter=bilateralFilter)

    for i, (imageLTensor, imageRTensor, ir_l_path, ir_r_path, depthTensor) in enumerate(DataLoader):
    # for i, (imageLTensor, imageRTensor, ir_l_path, ir_r_path) in enumerate(DataLoader):
        if torch.cuda.is_available():
            print("using cuda")
            imageLTensor = imageLTensor.cuda()
            imageRTensor = imageRTensor.cuda()
            depthTensor = depthTensor.cuda()
            beta = beta.cuda()
            sigmaColor = sigmaColor.cuda()
            sigmaSpace = sigmaSpace.cuda()
            f = f.cuda()
            baselineDis = baselineDis.cuda()

        # stereo matching
        depthImage, pointCloud = object(imageLTensor, imageRTensor, f, baselineDis, beta, sigmaColor, sigmaSpace)    # [B C H W]

        # # syndepth to pc
        # pointCloudSyn = DepthToPointCloud(depthTensor, f)

        # Plotting
        for i in range(imageLTensor.shape[0]):
            # save sim pc and depth
            # pointCloudCpu = pointCloud[i].data.cpu().numpy()
            # pointCloudCpu = np.delete(pointCloudCpu, pointCloudCpu[:, 2] < 0, axis=0)

            # pointCloudCaptra = copy.deepcopy(pointCloudCpu)
            # pointCloudCaptra[:, 0] = pointCloudCpu[:, 0]
            # pointCloudCaptra[:, 1] = pointCloudCpu[:, 1]
            # pointCloudCaptra[:, 2] = pointCloudCpu[:, 2]

            save_dir = os.path.join(save_root, ir_l_path[i].split("/")[-2])
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)

            # print(save_dir + "/%s_simDepthImage.png"%ir_l_path[i].split("/")[-1][:4])
            # plt.imsave(save_dir + "/%s_simDepthImage.png"%ir_l_path[i].split("/")[-1][:4], depthImage[i].squeeze().data.cpu().numpy()*255., cmap="gray")
            
            depthImageExr = depthImage[i].squeeze().data.cpu().numpy().astype("float32")
            depthImageExr = np.stack((depthImageExr,)*3, axis=-1)
            print(save_dir + "/%s_simDepthImage.exr"%ir_l_path[i].split("/")[-1][:4])
            cv2.imwrite(save_dir + "/%s_simDepthImage.exr"%ir_l_path[i].split("/")[-1][:4], depthImageExr)

            # print(save_dir + "/%s_simPointCloud.txt"%ir_l_path[i].split("/")[-1][:4])
            # np.savetxt(save_dir + "/%s_simPointCloud.txt"%ir_l_path[i].split("/")[-1][:4], pointCloudCaptra, fmt="%.04f", delimiter=" ")

            # # save syn pc and depth
            # pointCloudSynCpu = pointCloudSyn[i].data.cpu().numpy()
            # pointCloudSynCpu = np.delete(pointCloudSynCpu, pointCloudSynCpu[:, 2] < 0, axis=0)

            # pointCloudSynCaptra = copy.deepcopy(pointCloudSynCpu)
            # pointCloudSynCaptra[:, 0] = pointCloudSynCpu[:, 0]
            # pointCloudSynCaptra[:, 1] = pointCloudSynCpu[:, 1]
            # pointCloudSynCaptra[:, 2] = pointCloudSynCpu[:, 2]

            # plt.imsave(save_dir + "/%s_synDepthImage.png"%ir_l_path[i].split("/")[-1][:4], depthTensor[i].squeeze().data.cpu().numpy()*255., cmap="gray")
            # np.savetxt(save_dir + "/%s_synPointCloud.txt"%ir_l_path[i].split("/")[-1][:4], pointCloudSynCpu, fmt="%.04f", delimiter=" ")

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == "__main__":
    force_cudnn_initialization()
    main_batch()