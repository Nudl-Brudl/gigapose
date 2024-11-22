# https://github.com/bubbliiiing/stable-diffusion/blob/c9fd61042a80dd2239fcf13753c764cb08d692a9/ldm/modules/attention.py
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    A basic building block for ResNet architecture.

    This class defines a basic building block for 
    the ResNet architecture, containing two convolutional 
    layers with batch normalization and ReLU activation.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn1 (nn.BatchNorm2d): First batch normalization layer.
        bn2 (nn.BatchNorm2d): Second batch normalization layer.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential or None): Downsampling layer if stride > 1.
    """
    def __init__(self, in_planes, planes, stride=1):
        '''
        Initialize BasicBlock

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the first convolution. Default is 1.
        '''
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    '''
    Implements the Gated Exponential Linear Unit activation
    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    '''
    Implements a Linear layer with GEGLU
    '''

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention module.

    This class implements a spatial self-attention 
    mechanism using convolutions and softmax attention.

    Attributes:
        norm (nn.GroupNorm): Group normalization layer.
        q (nn.Conv2d): Query convolution.
        k (nn.Conv2d): Key convolution.
        v (nn.Conv2d): Value convolution.
        proj_out (nn.Conv2d): Output projection convolution.
    """

    def __init__(self, in_channels):
        '''
        Initialiazes SpatialSelfAttention

        Args:
            in_channels (int): Number of input channels.
        '''
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    """
    Cross-attention module.

    Attributes:
        scale (float): Scaling factor for attention scores.
        heads (int): Number of attention heads.
        to_q (nn.Linear): Linear layer for query projection.
        to_k (nn.Linear): Linear layer for key projection.
        to_v (nn.Linear): Linear layer for value projection.
        to_out (nn.Sequential): Output projection layers.
    """


    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        '''
        Initializes CrossAttention

        Args:
            query_dim (int): Dimension of the query.
            context_dim (int, optional): Dimension of the context. Defaults to None.
            heads (int): Number of attention heads. Default is 8.
            dim_head (int): Dimension of each attention head. Default is 64.
            dropout (float): Dropout rate. Default is 0.

    
        '''
        
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = sim.to(x.dtype)
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    '''
    Combine self-attention, cross-attention and feedforward layers
    '''

    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Spatial transformer module for image-like data.

    Attributes:
        in_channels (int): Number of input channels.
        norm (Normalize): Normalization layer.
        proj_in (nn.Conv2d or nn.Linear): Input projection layer.
        transformer_blocks (nn.ModuleList): List of transformer blocks.
        proj_out (nn.Conv2d or nn.Linear): Output projection layer.
        use_linear (bool): Whether linear layers are used instead of convolutions.
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=False):
        '''
        Init SpatialTransformer

        Args:
            in_channels (int): Number of input channels.
            n_heads (int): Number of attention heads.
            d_head (int): Dimension of each attention head.
            depth (int): Number of transformer blocks. Default is 1.
            dropout (float): Dropout rate. Default is 0.
            context_dim (int or list, optional): Dimension(s) of the context. Default is None.
            disable_self_attn (bool): Whether to disable self-attention. Default is False.
            use_linear (bool): Whether to use linear layers instead of convolutions. Default is False.
            use_checkpoint (bool): Whether to use checkpointing. Default is False.
        '''

        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class ResNet(nn.Module):
    """
    ResNet model with optional attention mechanisms.

    Attributes:
        use_attention (bool): Whether to use attention mechanisms.
        input_size (int): Input image size.
        in_planes (int): Number of input planes for each layer.
        conv1 (nn.Conv2d): Initial convolutional layer.
        bn1 (nn.BatchNorm2d): Initial batch normalization layer.
        relu (nn.ReLU): ReLU activation function.
        layer1, layer2, layer3, layer4 (nn.Sequential): ResNet layers.
        attention1, attention2 (SpatialTransformer): Optional attention modules.
        layer4_outconv (nn.Conv2d): Output convolution layer.
    """
    def __init__(self, config):
        '''
        Init ResNet

        Args:
            config (dict): Configuration dictionary containing model parameters.
        '''
        
        super().__init__()
        # Config
        block = BasicBlock
        self.use_attention = True if config["n_heads"]>0 else False
        input_dim = config["input_dim"]
        self.input_size = config["input_size"]
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]
        descriptor_size = config["descriptor_size"]
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(
            input_dim, initial_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16
        if self.use_attention:
            self.attention1 = SpatialTransformer(block_dims[1], n_heads=config["n_heads"], d_head=block_dims[1]//config["n_heads"], context_dim=block_dims[1])
            self.attention2 = SpatialTransformer(block_dims[3], n_heads=config["n_heads"], d_head=block_dims[3]//config["n_heads"], context_dim=block_dims[3])
        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], descriptor_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # dirty fix for (224, 224) input
        x = F.interpolate(
            x, (self.input_size, self.input_size), mode="bilinear", align_corners=True
        )
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        if self.use_attention:
            x2 = self.attention1(x2)
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16
        if self.use_attention:
            x4 = self.attention2(x4)
        x4_out = self.layer4_outconv(x4)

        return x4_out


if __name__ == "__main__":
    net = ResNet(
        config={
            "n_heads": 4,
            "input_dim": 3,
            "input_size": 256,
            "initial_dim": 128,
            "block_dims": [128, 256, 256, 512],
        }
    )

    img = torch.randn(2, 3, 224, 224)

    preds = net(img)  # (1, 1000)
    print(preds.shape)
