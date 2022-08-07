import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace)
        self.with_spectral_norm = with_spectral_norm
        official_padding_mode = ['zeros', 'circular']
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                #if self.with_explicit_padding:
                #    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm:
               x = self.bn(x)
            elif layer == 'act' and activate:
               x = self.activate(x)
        return x


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), num_classes=150, in_channels=[96, 192, 384, 768],
                 channels=512, dropout_ratio=0.1, align_corners=False, in_index=[0, 1, 2, 3], img_size = 512):
        super(UPerHead, self).__init__()#(input_transform='multiple_select', **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.input_transform = 'multiple_select'
        self.in_index = in_index
        self.img_size = img_size
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,    # in_channels
            self.channels,  # out_channels
            3,  # kernel_size
            padding=1  # kernel_size
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1)

        #self.fpn_bottleneck = nn.Sequential(

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(self.dropout_ratio)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs, input_org_shape):
        """Forward function."""

        inputs = self._transform_inputs(inputs)
        
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        ######################原始方案###########################
        # 直接在最后得到logits后做resize
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)

        output = self.cls_seg(output)   # [bs, 150, 128, 128]
        output = resize(
            input=output,
            size=(input_org_shape[0], input_org_shape[1]),
            mode='bilinear',
            align_corners=self.align_corners)
        ######################原始方案###########################

        ######################方案二#############################
        # 在fpn后seg_cls前resize
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     fpn_outs[i] = resize(
        #         fpn_outs[i],
        #         size=fpn_outs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # fpn_outs = torch.cat(fpn_outs, dim=1)
        # output = self.fpn_bottleneck(fpn_outs)

        # output = resize(
        #     input=output,
        #     size=(input_org_shape[0], input_org_shape[1]),
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        
        # output = self.cls_seg(output)   # [bs, 150, 128, 128]
        #########################################################

        ######################方案三#############################
        # 在fpn前resize
        # for i in range(used_backbone_levels - 1, -1, -1):
        #     fpn_outs[i] = resize(
        #         fpn_outs[i],
        #         # size=fpn_outs[0].shape[2:],
        #         size=(input_org_shape[0], input_org_shape[1]),
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # fpn_outs = torch.cat(fpn_outs, dim=1)
        # output = self.fpn_bottleneck(fpn_outs)        
        # output = self.cls_seg(output)   # [bs, 150, 128, 128]
        #########################################################
        return output


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 in_channels,
                 in_index,
                 channels,
                 num_classes,
                 align_corners=False,
                 dropout_ratio=0.5,

                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 img_size = 512):
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.dropout_ratio = dropout_ratio
        self.img_size = img_size

        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.input_transform = None
        super(FCNHead, self).__init__()
        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2)

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(self.dropout_ratio)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    # def normal_init(self, module, mean=0, std=1, bias=0):
    #     if hasattr(module, 'weight') and module.weight is not None:
    #         nn.init.normal_(module.weight, mean, std)
    #     if hasattr(module, 'bias') and module.bias is not None:
    #         nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def forward(self, inputs, input_org_shape):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)       # [bs, 150, 32, 32]

        # resize回原图尺寸
        output = resize(
            input=output,
            size=(input_org_shape[0], input_org_shape[1]),
            mode='bilinear',
            align_corners=self.align_corners)

        return output


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
        #nn.init.constant_(module.weight, 0)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)