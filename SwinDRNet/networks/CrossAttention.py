import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

# from swin_transformer import window_partition, window_reverse
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class WindowCrossAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table_1 = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table_2 = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_branch_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_branch_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop_branch_1 = nn.Dropout(attn_drop)
        self.attn_drop_branch_2 = nn.Dropout(attn_drop)

        self.proj_branch_1 = nn.Linear(dim, dim)
        self.proj_branch_2 = nn.Linear(dim, dim)

        self.proj_drop_branch_1 = nn.Dropout(proj_drop)
        self.proj_drop_branch_2 = nn.Dropout(proj_drop)

        # self.fuse = nn.Linear(dim * 2, dim)

        trunc_normal_(self.relative_position_bias_table_1, std=.02)
        trunc_normal_(self.relative_position_bias_table_2, std=.02)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        branch_1, branch_2 = x
        B_, N, C = branch_1.shape
        qkv_1 = self.qkv_branch_1(branch_1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_2 = self.qkv_branch_2(branch_2).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # make torchscript happy (cannot use tensor as tuple)
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]  # make torchscript happy (cannot use tensor as tuple)

        q_1 = q_1 * self.scale
        q_2 = q_2 * self.scale

        attn_1 = (q_2 @ k_1.transpose(-2, -1))
        attn_2 = (q_1 @ k_2.transpose(-2, -1))

        relative_position_bias_1 = self.relative_position_bias_table_1[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias_2 = self.relative_position_bias_table_2[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias_1 = relative_position_bias_1.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias_2 = relative_position_bias_2.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn_1 = attn_1 + relative_position_bias_1.unsqueeze(0)
        attn_2 = attn_2 + relative_position_bias_2.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn_1 = attn_1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_2 = attn_2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)

            attn_1 = attn_1.view(-1, self.num_heads, N, N)
            attn_2 = attn_2.view(-1, self.num_heads, N, N)

            attn_1 = self.softmax(attn_1)
            attn_2 = self.softmax(attn_2)
        else:
            attn_1 = self.softmax(attn_1)
            attn_2 = self.softmax(attn_2)

        attn_1 = self.attn_drop_branch_1(attn_1)
        attn_2 = self.attn_drop_branch_2(attn_2)

        branch_1 = (attn_1 @ v_1).transpose(1, 2).reshape(B_, N, C)
        branch_2 = (attn_2 @ v_2).transpose(1, 2).reshape(B_, N, C)

        branch_1 = self.proj_branch_1(branch_1)
        branch_2 = self.proj_branch_2(branch_2)

        branch_1 = self.proj_drop_branch_1(branch_1)
        branch_2 = self.proj_drop_branch_2(branch_2)

        x = [branch_1, branch_1]
        # x = torch.cat((branch_1, branch_2), -1)
        # x = self.fuse(x)
        return tuple(x)


class SwinCrossAttentionBlock(nn.Module):
    """ Swin Cross Attention Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_branch_1 = norm_layer(dim)
        self.norm1_branch_2 = norm_layer(dim)

        self.attn = WindowCrossAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_branch_1 = norm_layer(dim)
        self.norm2_branch_2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_branch_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_branch_2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """

        branch_1, branch_2 = x
        B, L, C = branch_1.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut_1 = branch_1
        shortcut_2 = branch_2
        branch_1 = self.norm1_branch_1(branch_1)
        branch_2 = self.norm1_branch_2(branch_2)
        branch_1 = branch_1.view(B, H, W, C)
        branch_2 = branch_2.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        branch_1 = F.pad(branch_1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        branch_2 = F.pad(branch_2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = branch_1.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_branch_1 = torch.roll(branch_1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_branch_2 = torch.roll(branch_2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_branch_1 = branch_1
            shifted_branch_2 = branch_2
            attn_mask = None

        # partition windows
        x_windows_branch_1 = window_partition(shifted_branch_1, self.window_size)  # nW*B, window_size, window_size, C
        x_windows_branch_2 = window_partition(shifted_branch_2, self.window_size)  # nW*B, window_size, window_size, C

        x_windows_branch_1 = x_windows_branch_1.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows_branch_2 = x_windows_branch_2.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        x_windows = tuple([x_windows_branch_1, x_windows_branch_2])
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        attn_windows_branch_1, attn_windows_branch_2 = attn_windows

        # merge windows
        attn_windows_branch_1 = attn_windows_branch_1.view(-1, self.window_size, self.window_size, C)
        attn_windows_branch_2 = attn_windows_branch_2.view(-1, self.window_size, self.window_size, C)

        shifted_x_branch_1 = window_reverse(attn_windows_branch_1, self.window_size, Hp, Wp)  # B H' W' C
        shifted_x_branch_2 = window_reverse(attn_windows_branch_2, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            branch_1 = torch.roll(shifted_x_branch_1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            branch_2 = torch.roll(shifted_x_branch_2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            branch_1 = shifted_branch_1
            branch_2 = shifted_branch_2

        if pad_r > 0 or pad_b > 0:
            branch_1 = branch_1[:, :H, :W, :].contiguous()
            branch_2 = branch_2[:, :H, :W, :].contiguous()

        branch_1 = branch_1.view(B, H * W, C)
        branch_2 = branch_2.view(B, H * W, C)

        # FFN
        branch_1 = shortcut_1 + self.drop_path(branch_1)
        branch_2 = shortcut_2 + self.drop_path(branch_1)

        branch_1 = branch_1 + self.drop_path(self.mlp_branch_1(self.norm2_branch_1(branch_1)))
        branch_2 = branch_2 + self.drop_path(self.mlp_branch_2(self.norm2_branch_2(branch_2)))

        return tuple([branch_1, branch_2])


class BasicCrossAttentionLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinCrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.fuse = nn.Linear(dim*2, dim)
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x[0].device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        

        branch_1, branch_2 = x

        x = torch.cat((branch_1, branch_2), -1)
        x = self.fuse(x)
        # x = branch_2

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class CrossAttention(nn.Module):
    def __init__(self, 
                 in_channel=96,      # feature map dim
                 depth=2,                 
                 num_heads=3):
        super().__init__()
        self.cross_attn_layer = BasicCrossAttentionLayer(dim=in_channel, depth=depth, num_heads=num_heads)
        self.num_feature = in_channel

        # add a norm layer for each output
        self.norm_layer = nn.LayerNorm(in_channel)


    def forward(self, x):
        rgb_feature_map, xyz_feature_map = x
        norm_layer = self.norm_layer
        cross_attn_layer = self.cross_attn_layer
        
        B, C, H, W = rgb_feature_map.shape
        rgb_feature_map = rgb_feature_map.permute(0, 2, 3, 1).view(B, H*W, C)
        xyz_feature_map = xyz_feature_map.permute(0, 2, 3, 1).view(B, H*W, C)

        x = tuple([rgb_feature_map, xyz_feature_map])
        x_out, H, W, x, Wh, Ww = cross_attn_layer(x, H, W)
        x_out = norm_layer(x_out)
        out = x_out.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        return out


    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)