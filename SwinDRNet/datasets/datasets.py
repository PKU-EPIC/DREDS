#!/usr/bin/env python3

import os
import glob
import sys
from PIL import Image
import Imath
import numpy as np
import OpenEXR

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
from tqdm import tqdm
import cv2
import copy


def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array

    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)

    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr



augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({
        "height": 224,
        "width": 224
    }, interpolation='cubic'),

    # iaa.Resize({
    #     "height": 224,
    #     "width": 224
    # }, interpolation='nearest'),
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),

    # Bright Patches
    iaa.Sometimes(
        0.1,
        iaa.blend.Alpha(factor=(0.2, 0.7),
                        first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                          upscale_method='cubic',
                                                          iterations=(1, 2)),
                        name="simplex-blend")),

    # Color Space Mods
    iaa.Sometimes(
        0.3,
        iaa.OneOf([
            iaa.Add((20, 20), per_channel=0.7, name="add"),
            iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                               name="hue"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                               name="sat"),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
            iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
        ])),

    # Blur and Noise
    iaa.Sometimes(
        0.2,
        iaa.SomeOf((1, None), [
            iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                       iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
            iaa.OneOf([
                iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
            ]),
        ],
                   random_order=True)),

    # Colored Blocks
    iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
])

# Validation Dataset
augs_test = iaa.Sequential([
    iaa.Resize({
        "height": 224,
        "width": 224
    }, interpolation='nearest'),
])


input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]


def load_meta(file_path):
    with open(file_path, "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
    meta = []
    for line in data:
        words = line.split(" ")
        instance = {}
        if len(words) > 5:
            instance["index"] = int(words[0])
            instance["label"] = int(words[1])
            instance["instance_folder"] = words[2]
            instance["name"] = words[3]
            instance["scale"] = float(words[4])
            instance["material"] = int(words[5])
            instance["quaternion"] = np.array([float(words[6]), float(words[7]), float(words[8]), float(words[9])])
            instance["translation"] = np.array([float(words[10]), float(words[11]), float(words[12])])
        else:
            instance["index"] = int(words[0])
            instance["label"] = int(words[1])
            instance["instance_folder"] = -1
            instance["name"] = -1
            instance["scale"] = -1
            instance["material"] = int(words[2])
            instance["quaternion"] = np.array([0., 0., 0., 0.])
            instance["translation"] = np.array([0., 0., 0.])
        meta.append(instance)
        
    while len(meta) < 30:
        instance = {}
        instance["index"] = -1
        instance["label"] = -1
        instance["instance_folder"] = " "
        instance["name"] = " "
        instance["scale"] = -1.
        instance["material"] = -1 
        instance["quaternion"] = np.array([0., 0., 0., 0.])
        instance["translation"] = np.array([0., 0., 0.])
        meta.append(instance)
    return meta


class SwinDRNetDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.
    """

    def __init__(
            self,
            fx=446.31,
            fy=446.31,
            rgb_dir='',
            sim_depth_dir='',
            syn_depth_dir='',
            nocs_dir='',
            mask_dir='',
            meta_dir='',
            transform=None,
            input_only=None,
            material_valid={'transparent': True,
                            'specular': True,
                            'diffuse': False}
    ):

        super().__init__()
        self.rgb_dir = rgb_dir
        self.sim_depth_dir = sim_depth_dir
        self.syn_depth_dir = syn_depth_dir
        self.nocs_dir = nocs_dir
        self.mask_dir = mask_dir
        self.meta_dir = meta_dir
        self.fx = fx
        self.fy = fy

        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_rgb = []
        self._datalist_sim_depth = []  # Variable containing list of all input images filenames in dataset
        self._datalist_syn_depth = []
        self._datalist_nocs = []
        self._datalist_mask = []
        self._datalist_meta = []

        self._extension_rgb = ['_color.png']
        self._extension_sim_depth = ['_simDepthImage.exr', '_depth_415.exr']  # The file extension of input images
        self._extension_syn_depth = ['_depth_120.exr', '_gt_depth.exr', '_depth_0.exr']
        self._extension_nocs = ['_coord.png']
        self._extension_mask = ['_mask.exr', '_mask.png']
        self._extension_meta = ['_meta.txt']
        
        self._create_lists_filenames(self.rgb_dir,
                                     self.sim_depth_dir,
                                     self.syn_depth_dir,
                                     self.nocs_dir,
                                     self.mask_dir,
                                     self.meta_dir)
        
        # material mask
        self.mask_transparent = material_valid['transparent']
        self.mask_specular = material_valid['specular']
        self.mask_diffuse = material_valid['diffuse']

    def __len__(self):
        return len(self._datalist_rgb)


    def __getitem__(self, index):
        '''
        Returns an item from the dataset at the given index. 
        '''
        
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        # Open rgb images
        rgb_path = self._datalist_rgb[index]
        _rgb = Image.open(rgb_path).convert('RGB')
        _rgb = np.array(_rgb)

        # Open simulated depth images
        sim_depth_path = self._datalist_sim_depth[index]
        _sim_depth =  cv2.imread(sim_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
        if len(_sim_depth.shape) == 3:
            _sim_depth = _sim_depth[:, :, 0]
        _sim_depth = _sim_depth[np.newaxis, ...]
        
        # Open synthetic depth images
        syn_depth_path = self._datalist_syn_depth[index]
        _syn_depth =  cv2.imread(syn_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(_syn_depth.shape) == 3:
            _syn_depth = _syn_depth[:, :, 0]
        _syn_depth = _syn_depth[np.newaxis, ...]
        
        # Open nocs images
        nocs_path = self._datalist_nocs[index]
        _nocs = Image.open(nocs_path).convert('RGB')
        _nocs = np.array(_nocs) / 255.

        # Open mask images
        mask_path = self._datalist_mask[index]

        if mask_path.split('.')[-1] == 'exr':
            _mask = exr_loader(mask_path, ndim=1)
        else:
            _mask = Image.open(mask_path)
            _mask = np.array(_mask)

        # Open meta files
        meta_path = self._datalist_meta[index]
        _meta = load_meta(meta_path)

        ori_h = _sim_depth.shape[1]
        ori_w = _sim_depth.shape[2]
        
        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            det_tf_only_resize = augs_test.to_deterministic()

            _sim_depth = _sim_depth.transpose((1, 2, 0))  # To Shape: (H, W, 1)
            # transform to xyz_img
            img_h = _sim_depth.shape[0]
            img_w = _sim_depth.shape[1]
            
            _sim_depth = det_tf_only_resize.augment_image(_sim_depth, hooks=ia.HooksImages(activator=self._activator_masks))
            _sim_depth = _sim_depth.transpose((2, 0, 1))  # To Shape: (1, H, W)
            _sim_depth[_sim_depth <= 0] = 0.0
            _sim_depth = _sim_depth.squeeze(0)            # (H, W)

            fx = self.fx
            fy = self.fy
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
            scale = (224 / img_w, 224 / img_h)
            camera_params['fx'] *= scale[0]
            camera_params['fy'] *= scale[1]
            # camera_params['cx'] *= scale[0]
            # camera_params['cy'] *= scale[1]
            camera_params['cx'] = 112 - 0.5
            camera_params['cy'] = 112 - 0.5
            camera_params['xres'] *= scale[0]
            camera_params['yres'] *= scale[1]

            _sim_xyz = self.compute_xyz(_sim_depth, camera_params)            
            _rgb = det_tf.augment_image(_rgb)

            _syn_depth = _syn_depth.transpose((1, 2, 0))  # To Shape: (H, W, 1)
            _syn_depth = det_tf_only_resize.augment_image(_syn_depth, hooks=ia.HooksImages(activator=self._activator_masks))
            _syn_depth = _syn_depth.transpose((2, 0, 1))  # To Shape: (1, H, W)

            _nocs = det_tf_only_resize.augment_image(_nocs, hooks=ia.HooksImages(activator=self._activator_masks))
            _nocs = _nocs.transpose((2, 0, 1))
            _mask = det_tf_only_resize.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _rgb_tensor = transforms.ToTensor()(_rgb)
        _sim_xyz_tensor = transforms.ToTensor()(_sim_xyz)
        _sim_depth_tensor = transforms.ToTensor()(_sim_depth)
        _syn_depth_tensor = torch.from_numpy(_syn_depth)
        _nocs_tensor = torch.from_numpy(_nocs)
        

        if mask_path.split('.')[-1] == 'exr':
            _mask = np.array(_mask * 255, dtype=np.int32)        

        # transform to matrial semantic mask
        _mask_material = np.full(_mask.shape, -1) #, 0）
        for i in range(len(_meta)):
            _mask_material[_mask == _meta[i]["index"]] = _meta[i]["material"] #1
        
        _mask_ins = np.full(_mask.shape, 1)
        if self.mask_transparent:
            _mask_ins[_mask_material == 2] = 0
        if self.mask_specular:
            _mask_ins[_mask_material == 3] = 0
        if self.mask_diffuse:
            _mask_ins[_mask_material == 0] = 0
            _mask_ins[_mask_material == 1] = 0

        # semantatic segmentation mask
        _mask_sem = np.full(_mask.shape, 8) #, 0）
        for i in range(len(_meta)):
            _mask_sem[_mask == _meta[i]["index"]] = _meta[i]["label"]

        _mask_ins_without_other = np.full(_mask.shape, 0)
        _mask_ins_without_other[_mask == 255] = 1
        _mask_ins_without_other[_mask_sem == 0] = 1

        _mask_sem = _mask_sem[..., np.newaxis]
        _mask_ins = _mask_ins[..., np.newaxis]
        _mask_ins_without_other = _mask_ins_without_other[..., np.newaxis]

        _mask_sem_tensor = transforms.ToTensor()(_mask_sem)
        _mask_ins_tensor = transforms.ToTensor()(_mask_ins)
        _mask_ins_without_other_tensor = transforms.ToTensor()(_mask_ins_without_other)
        
        _mask_ins_tensor = 1 - _mask_ins_tensor
        _mask_ins_without_other_tensor = 1 - _mask_ins_without_other_tensor
              
                    
        # cache = (_rgb_tensor, _sim_xyz_tensor, _sim_depth_tensor, _syn_depth_tensor, _surface_normal_tensor, \
        #        _nocs_tensor, _mask_sem_tensor, _mask_ins_tensor, _mask_ins_without_other_tensor)
               
        cache = (_rgb_tensor, _sim_xyz_tensor, _sim_depth_tensor, _syn_depth_tensor, \
               _nocs_tensor, _mask_sem_tensor, _mask_ins_tensor, _mask_ins_without_other_tensor)
        sample = {'rgb': _rgb_tensor,
                  'sim_xyz': _sim_xyz_tensor,
                  'sim_depth': _sim_depth_tensor,
                  'syn_depth': _syn_depth_tensor,
                  'nocs_map': _nocs_tensor,
                  'sem_mask': _mask_sem_tensor,
                  'ins_mask': _mask_ins_tensor,
                  'ins_w/o_others_mask': _mask_ins_without_other}

        # return cache
        return sample


    def _create_lists_filenames(self, rgb_dir, sim_depth_dir, syn_depth_dir, nocs_dir, mask_dir, meta_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored
            labels_dir (str): Path to the dir where masks are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        # generate rgb iamge paths list
        assert os.path.isdir(rgb_dir), 'Dataloader given rgbs directory that does not exist: "%s"' % (rgb_dir)
        for ext in self._extension_rgb:
            rgb_search_str = os.path.join(rgb_dir, '*/*' + ext)
            rgb_paths = sorted(glob.glob(rgb_search_str))
            self._datalist_rgb = self._datalist_rgb + rgb_paths
        num_rgb = len(self._datalist_rgb)
        if num_rgb == 0:
            raise ValueError('No RGBs found in given directory. Searched in dir: {} '.format(rgb_search_str))


        # generate simulation depth iamge paths list
        assert os.path.isdir(sim_depth_dir), 'Dataloader given simulation depth images directory that does not exist: "%s"' % (sim_depth_dir)
        for ext in self._extension_sim_depth:
            sim_depth_search_str = os.path.join(sim_depth_dir, '*/*' + ext)
            sim_depth_paths = sorted(glob.glob(sim_depth_search_str))
            self._datalist_sim_depth = self._datalist_sim_depth + sim_depth_paths
        num_sim_depth = len(self._datalist_sim_depth)
        if num_sim_depth == 0:
            raise ValueError('No simulation depth images found in given directory. Searched in dir: {} '.format(sim_depth_search_str))
        if num_sim_depth != num_rgb:
            raise ValueError('The number of simulation depth images and rgb images do not match. Please check data,' +
                                'found {} simulation depth images and {} rgb images in dirs:\n'.format(num_sim_depth, num_rgb) +
                                'simulation depth images: {}\nrgb images: {}\n'.format(syn_depth_dir, rgb_dir))


        # generate synthetic depth image paths list
        assert os.path.isdir(syn_depth_dir), ('Dataloader given synthetic depth images directory that does not exist: "%s"' %
                                            (syn_depth_dir))
        for ext in self._extension_syn_depth:
            syn_depth_search_str = os.path.join(syn_depth_dir, '*/*' + ext)
            syn_depth_paths = sorted(glob.glob(syn_depth_search_str))
            self._datalist_syn_depth = self._datalist_syn_depth + syn_depth_paths

        num_syn_depth = len(self._datalist_syn_depth)
        if num_syn_depth == 0:
            raise ValueError('No synthetic depth images found in given directory. Searched for {}'.format(syn_depth_search_str))
        if num_syn_depth != num_rgb:
            raise ValueError('The number of synthetic depth images and rgb images do not match. Please check data,' +
                                'found {} synthetic depth images and {} rgb images in dirs:\n'.format(num_syn_depth, num_rgb) +
                                'synthetic depth images: {}\nrgb images: {}\n'.format(syn_depth_dir, rgb_dir))


        # generate nocs image paths list
        assert os.path.isdir(nocs_dir), ('Dataloader given nocs iamges directory that does not exist: "%s"' %
                                            (nocs_dir))
        for ext in self._extension_nocs:
            nocs_search_str = os.path.join(nocs_dir, '*/*' + ext)
            nocs_paths = sorted(glob.glob(nocs_search_str))
            self._datalist_nocs = self._datalist_nocs + nocs_paths

        num_nocs = len(self._datalist_nocs)

        if num_nocs == 0:
            nocs_search_str = os.path.join(nocs_dir, '*/*' + '_color.png')
            nocs_paths = sorted(glob.glob(nocs_search_str))
            self._datalist_nocs = self._datalist_nocs + nocs_paths
        num_nocs = len(self._datalist_nocs)    
        
        if num_nocs == 0:
            raise ValueError('No nocs images found in given directory. Searched for {}'.format(nocs_search_str))
        if num_nocs != num_rgb:
            raise ValueError('The number of nocs images and rgb images do not match. Please check data,' +
                                'found {} nocs images and {} rgb images in dirs:\n'.format(num_nocs, num_rgb) +
                                'nocs images: {}\nrgb images: {}\n'.format(nocs_dir, rgb_dir))


        # generate mask image paths list
        assert os.path.isdir(mask_dir), ('Dataloader given mask iamges directory that does not exist: "%s"' %
                                            (mask_dir))
        for ext in self._extension_mask:
            mask_search_str = os.path.join(mask_dir, '*/*' + ext)
            mask_paths = sorted(glob.glob(mask_search_str))
            self._datalist_mask = self._datalist_mask + mask_paths

        num_mask = len(self._datalist_mask)
        if num_mask == 0:
            raise ValueError('No mask images found in given directory. Searched for {}'.format(mask_search_str))
        if num_mask != num_rgb:
            raise ValueError('The number of mask images and rgb images do not match. Please check data,' +
                                'found {} mask images and {} rgb images in dirs:\n'.format(num_mask, num_rgb) +
                                'mask images: {}\nrgb images: {}\n'.format(mask_dir, rgb_dir))



        assert os.path.isdir(meta_dir), ('Dataloader given metas directory that does not exist: "%s"' %
                                            (meta_dir))
        for ext in self._extension_meta:
            meta_search_str = os.path.join(meta_dir, '*/*' + ext)
            meta_paths = sorted(glob.glob(meta_search_str))
            self._datalist_meta = self._datalist_meta + meta_paths

        num_meta = len(self._datalist_meta)
        if num_meta == 0:
            raise ValueError('No metas found in given directory. Searched for {}'.format(meta_search_str))
        if num_meta != num_rgb:
            raise ValueError('The number of metas and rgb images do not match. Please check data,' +
                                'found {} metas and {} rgb images in dirs:\n'.format(num_meta, num_rgb) +
                                'metas: {}\nrgb images: {}\n'.format(meta_dir, rgb_dir))

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default    

    def compute_xyz(self, depth_img, camera_params):
        """ Compute ordered point cloud from depth image and camera parameters.

            If focal lengths fx,fy are stored in the camera_params dictionary, use that.
            Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

            @param depth_img: a [H x W] numpy array of depth values in meters
            @param camera_params: a dictionary with parameters of the camera used
        """
        # Compute focal length from camera parameters
        fx = camera_params['fx']
        fy = camera_params['fy']
        x_offset = camera_params['cx']
        y_offset = camera_params['cy']
        indices = np.indices((int(camera_params['yres']), int(camera_params['xres'])), dtype=np.float32).transpose(1,2,0)
        z_e = depth_img
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
        return xyz_img