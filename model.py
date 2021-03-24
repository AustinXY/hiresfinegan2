import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from tqdm.std import trange

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor,
                        down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1,
                        down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        w_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(
                pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(w_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(
                    0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(
                    input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        w_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            w_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, w_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, 3, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

#----------------------------------------------------------------------------

class MappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim,          # Input latent (Z) dimensionality.
        w_dim,          # Intermediate latent (W) dimensionality.
        n_mlp,          # Number of mapping layers.
        lr_mlp = 0.01,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        layers = [PixelNorm()]

        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    z_dim, w_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self, z, truncation=1, truncation_latent=None):
        # Main layers.
        x = self.style(z)

        # Apply truncation.
        if truncation < 1:
            x = truncation_latent + truncation * (x - truncation_latent)

        return x

#----------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(
        self,
        img_resolution,
        z_dim,                              # Input latent (Z) dimensionality.
        w_dim,                              # Intermediate latent (W) dimensionality.
        n_mlp,                              # Number of mapping layers.
        mix_prob           = 0.9,           # style mixing probability.
        channel_multiplier = 2,
        blur_kernel        = [1, 3, 3, 1],
        lr_mlp             = 0.01,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.w_dim = w_dim
        self.z_dim = z_dim
        self.mix_prob = mix_prob

        self.log_size = int(math.log(img_resolution, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_ws = self.log_size * 2 - 2
        self.mapping = MappingNetwork(z_dim, w_dim, n_mlp, lr_mlp=lr_mlp)
        self.img_mapping = ImgMappingNetwork(self.num_ws, w_dim)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, w_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], w_dim, upsample=False)

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(
                f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    w_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, w_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, w_dim))

            in_channel = out_channel

    # def make_noise(self):
    #     device = self.input.input.device

    #     noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

    #     for i in range(3, self.log_size + 1):
    #         for _ in range(2):
    #             noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    #     return noises

    # def mean_latent(self, n_latent):
    #     latent_in = torch.randn(
    #         n_latent, self.w_dim, device=self.input.input.device
    #     )
    #     latent = self.mapping(latent_in).mean(0, keepdim=True)

    #     return latent

    # def get_latent(self, input):
    #     return self.mapping(input)

    def style_mixing(self, z, mix_styles=True, truncation=1, truncation_latent=None):
        latent = self.mapping(z, truncation=truncation, truncation_latent=truncation_latent)
        if mix_styles and self.mix_prob > 0 and random.random() < self.mix_prob:
            z2 = torch.randn(z.size(0), self.z_dim, device=z.device)
            latent2 = self.mapping(z2, truncation=truncation, truncation_latent=truncation_latent)
            inject_index = random.randint(1, self.num_ws - 1)
            latent = latent.unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = latent2.unsqueeze(1).repeat(1, self.num_ws - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)
        else:
            latent = latent.unsqueeze(1).repeat(1, self.num_ws, 1)
        return latent

    def forward(
        self,
        z = None,
        fine_img = None,
        return_latents=False,
        mix_styles=True,
        noise=None,
        randomize_noise=True,
        truncation=1,
        truncation_latent=None,
    ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]
        if z is not None:
            latent = self.style_mixing(z, mix_styles, truncation=truncation, truncation_latent=truncation_latent)
        else:
            assert fine_img is not None
            latent = self.img_mapping(fine_img)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4],
                        activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

#----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvLayer(in_channel, out_channel, 3, activate=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            ConvLayer(out_channel, out_channel, 3, activate=False, downsample=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out


class ImgMappingNetwork(nn.Module):
    def __init__(self, num_ws, w_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        self.inc = nn.Sequential(
            ConvLayer(3, 64, 3, activate=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.down5 = DoubleConv(512, 512)

        self.final_linear = nn.Sequential(
            EqualLinear(512 * 4 * 4, num_ws * w_dim),
            nn.BatchNorm1d(num_ws * w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(num_ws * w_dim, num_ws * w_dim),
            nn.BatchNorm1d(num_ws * w_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        x = self.inc(input)
        x = self.down1(x)  # 64 x 64
        x = self.down2(x)  # 32 x 32
        x = self.down3(x)  # 16 x 16
        x = self.down4(x)  # 8 x 8
        x = self.down5(x)  # 4 x 4
        # x = self.down6(x)  # 1 x 1
        x = self.final_linear(x.view(x.size(0), -1))
        return x.view(x.size(0), self.num_ws, self.w_dim)


#----------------------------------------------------------------------------

# ############## FINE_G networks ################################################
# Upsale the spatial size by a factor of 2
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def convlxl(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
                     padding=1, bias=False)

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class _ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(_ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, c_flag, z_dim=100, p_dim=20, c_dim=200):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.c_flag= c_flag

        if self.c_flag==1 :
            self.in_dim = z_dim + p_dim
        elif self.c_flag==2:
            self.in_dim = z_dim + c_dim

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 16)


    def forward(self, z_code, code):

        in_code = torch.cat((code, z_code), 1)
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, use_hrc=1, num_residual=2, p_dim=20, c_dim=200):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if use_hrc == 1: # For parent stage
            self.ef_dim = p_dim
        else:            # For child stage
            self.ef_dim = c_dim

        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(_ResBlock, ngf)
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code



class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img



class GET_MASK_G(nn.Module):
    def __init__(self, ngf):
        super(GET_MASK_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self, gf_dim=64):
        super(G_NET, self).__init__()
        self.gf_dim = gf_dim
        self.define_module()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.scale_fimg = nn.UpsamplingBilinear2d(size = [126, 126])

    def define_module(self):
        #Background stage
        self.h_net1_bg = INIT_STAGE_G(self.gf_dim * 16, 2)
        self.img_net1_bg = GET_IMAGE_G(self.gf_dim) # Background generation network

        # Parent stage networks
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, 1)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc = 1)
        self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)  # Parent foreground generation network
        self.img_net2_mask= GET_MASK_G(self.gf_dim // 2) # Parent mask generation network

        # Child stage networks
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, use_hrc = 0)
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 4) # Child foreground generation network
        self.img_net3_mask = GET_MASK_G(self.gf_dim // 4) # Child mask generation network

    def forward(self, z_code, bg_code, p_code, c_code):
        fake_imgs = [] # Will contain [background image, parent image, child image]
        fg_imgs = [] # Will contain [parent foreground, child foreground]
        mk_imgs = [] # Will contain [parent mask, child mask]
        fg_mk = [] # Will contain [masked parent foreground, masked child foreground]

        #Background stage
        h_code1_bg = self.h_net1_bg(z_code, bg_code)
        fake_img1 = self.img_net1_bg(h_code1_bg) # Background image
        fake_img1_126 = self.scale_fimg(fake_img1) # Resizing fake background image from 128x128 to the resolution which background discriminator expects: 126 x 126.
        fake_imgs.append(fake_img1_126)

        #Parent stage
        h_code1 = self.h_net1(z_code, p_code)
        h_code2 = self.h_net2(h_code1, p_code)
        fake_img2_foreground = self.img_net2(h_code2) # Parent foreground
        fake_img2_mask = self.img_net2_mask(h_code2) # Parent mask
        ones_mask_p = torch.ones_like(fake_img2_mask)
        opp_mask_p = ones_mask_p - fake_img2_mask
        fg_masked2 = torch.mul(fake_img2_foreground, fake_img2_mask)
        fg_mk.append(fg_masked2)
        bg_masked2 = torch.mul(fake_img1, opp_mask_p)
        fake_img2_final = fg_masked2 + bg_masked2 # Parent image
        fake_imgs.append(fake_img2_final)
        fg_imgs.append(fake_img2_foreground)
        mk_imgs.append(fake_img2_mask)

        #Child stage
        h_code3 = self.h_net3(h_code2, c_code)
        fake_img3_foreground = self.img_net3(h_code3) # Child foreground
        fake_img3_mask = self.img_net3_mask(h_code3) # Child mask
        ones_mask_c = torch.ones_like(fake_img3_mask)
        opp_mask_c = ones_mask_c - fake_img3_mask
        fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask)
        fg_mk.append(fg_masked3)
        bg_masked3 = torch.mul(fake_img2_final, opp_mask_c)
        fake_img3_final = fg_masked3 + bg_masked3  # Child image
        fake_imgs.append(fake_img3_final)
        fg_imgs.append(fake_img3_foreground)
        mk_imgs.append(fake_img3_mask)

        return fake_img3_final
        # return fake_imgs, fg_imgs, mk_imgs, fg_mk


#----------------------------------------------------------------------------