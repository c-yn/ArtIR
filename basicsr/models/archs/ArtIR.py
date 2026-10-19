import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
##########################################################################
## FFN
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Large kernel module
class LargeKernel(nn.Module):
    def __init__(self, dim, k):
        super(LargeKernel, self).__init__()

        self.k = k

        self.w = nn.Conv2d(1, k**2, kernel_size=1)
        self.pad = nn.ReflectionPad2d(k//2)
        self.a = nn.Parameter(torch.ones(dim, 1, 1))

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, dim, kernel_size=1),
        )

    def forward(self, x):
        one_channel = torch.mean(x, dim=1, keepdim=True)
        n, _, h, w = one_channel.shape
        weights = self.w(one_channel).unsqueeze(1) 
        one_channel2 = F.unfold(self.pad(one_channel), kernel_size=self.k).reshape(n, 1, self.k**2, h, w)
        x_ = torch.sum(one_channel2*weights, dim=2)

        return x * x_ * self.sca(one_channel)

class HighOrder(nn.Module):
    def __init__(self, dim, order, k, bias):
        super(HighOrder, self).__init__()

        self.order = order
        self.proj_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        dw_dim = sum(self.dims)
        self.dw = nn.Conv2d(dw_dim, dw_dim, kernel_size=3, stride=1, padding=1, groups=dw_dim, bias=bias)

        self.ks = [k // 2**i for i in range(order)]
        self.ks.reverse()
        self.lks = nn.ModuleList([LargeKernel(self.dims[i], self.ks[i]) for i in range(order)])

        self.projs = nn.ModuleList([nn.Conv2d(self.dims[i+1], self.dims[i+1], kernel_size=7, padding=3, groups=self.dims[i+1]) for i in range(order-1)])


    def forward(self, x):
        x = self.proj_in(x)
        y, x = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)
        x = self.dw(x)
        x_list = torch.split(x, self.dims, dim=1)
        y = self.lks[0](y)
        x = y * x_list[0]

        for i in range(self.order - 1):
            x = x.repeat_interleave(2, dim=1)
            
            x = self.projs[i](x)
            x = self.lks[i+1](x)
            x = x * x_list[i+1]
        return x


class AttModule(nn.Module):
    def __init__(self, dim, bias, k):
        super(AttModule, self).__init__()
        order = 3

        self.k_l = k
        self.dim = dim

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        if self.k_l != 0:
            self.HO = HighOrder(dim, order=order, k=k, bias=bias)
        else:
            self.to_hidden = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
            self.to_hidden_dw = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

    def forward(self, x):

        if self.k_l != 0:
            out = self.HO(x)
        else:
            x = self.to_hidden(x)
            x1, x2 = self.to_hidden_dw(x).chunk(2, dim=1)

            out = x1 * x2
        out = self.project_out(out)

        return out

##########################################################################
class AttBlock(nn.Module):
    def __init__(self, dim, k=0, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(AttBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttModule(dim, bias, k)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.conv = nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- ArtIR -----------------------
class ArtIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[1,1,1],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(ArtIR, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            AttBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=0) 
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=0) 
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=63) 
            for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=63) 
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=0) 
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            AttBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=0) 
            for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            AttBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias, k=0) 
            for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
    
