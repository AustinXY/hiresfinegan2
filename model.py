import math
from os import times
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


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
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

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
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

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
        out = F.conv2d(
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
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
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

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

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

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

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
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
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
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
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
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 4, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 4, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class Stage_Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        code0_len=0,
        code1_len=0,
        stage0_depth=-1,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        ):
        super().__init__()

        self.size = size
        self.stage0_depth = stage0_depth
        self.style_dim = style_dim

        code_channels = []
        for i in range(0, 9):
            cc = code0_len
            if i > stage0_depth:
                cc = code1_len
            code_channels.append(cc)

        self.channels = {
            4: 512 + code_channels[0],
            8: 512 + code_channels[1],
            16: 512 + code_channels[2],
            32: 512 + code_channels[3],
            64: 256 * channel_multiplier + code_channels[4],
            128: 128 * channel_multiplier + code_channels[5],
            256: 64 * channel_multiplier + code_channels[6],
            512: 32 * channel_multiplier + code_channels[7],
            1024: 16 * channel_multiplier + code_channels[8],
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim+code_channels[0], blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim+code_channels[0], upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim+code_channels[i-2],
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim+code_channels[i-2], blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim+code_channels[i-2]))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    # def mean_latent(self, n_latent):
    #     latent_in = torch.randn(
    #         n_latent, self.style_dim, device=self.input.input.device
    #     )
    #     latent = self.style(latent_in).mean(0, keepdim=True)

    #     return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        latent,
        code0,
        code1,
        noise=None,
        randomize_noise=True,
        ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        skip_li = []

        depth = 0
        code = code0
        if depth > self.stage0_depth:
            code = code1

        # print(code.size())
        # print(latent[:, 0].size())
        # sys.exit()

        _latent0 = torch.cat([latent[:, 0], code], 1)
        _latent1 = torch.cat([latent[:, 1], code], 1)

        out = self.input(latent)
        out = self.conv1(out, _latent0, noise=noise[0])
        skip = self.to_rgb1(out, _latent1)

        skip_li.append(skip)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):

            depth += 1
            if depth > self.stage0_depth:
                code = code1

            _latent0 = torch.cat([latent[:, i], code], 1)
            _latent1 = torch.cat([latent[:, i+1], code], 1)
            _latent2 = torch.cat([latent[:, i+2], code], 1)

            out = conv1(out, _latent0, noise=noise1)
            out = conv2(out, _latent1, noise=noise2)
            skip = to_rgb(out, _latent2, skip)
            skip_li.append(skip)

            i += 2

        return skip_li


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        p_categories,
        c_categories,
        p_size=None,
        b_categories=None,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        layers = [PixelNorm()]

        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        if b_categories is None:
            b_categories = c_categories

        self.b_categories = b_categories
        self.p_categories = p_categories
        self.c_categories = c_categories
        if p_size is None:
            p_size = size // 2

        self.p_depth = int(math.log(p_size, 2)) - 2

        self.bg_generator = Stage_Generator(size, style_dim, code1_len=self.b_categories,
            channel_multiplier=channel_multiplier, blur_kernel=blur_kernel)
        self.fg_generator = Stage_Generator(size, style_dim, code0_len=p_categories,
            code1_len=c_categories, stage0_depth=self.p_depth, channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = self.log_size * 2 - 2


    def child_to_parent(self, c_code):
        ratio = self.c_categories / self.p_categories
        cid = torch.argmax(c_code,  dim=1)
        pid = (cid / ratio).long()
        p_code = torch.zeros([c_code.size(0), self.p_categories]).cuda()
        for i in range(c_code.size(0)):
            p_code[i][pid[i]] = 1
        return p_code

    def forward(
        self,
        styles,
        c_code,
        tied_code=True,
        b_code=None,
        p_code=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        ):


        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if tied_code:
            p_code = self.child_to_parent(c_code)
            b_code = c_code

        bg_skip_li = self.bg_generator(latent, None, b_code, noise=noise, randomize_noise=randomize_noise)

        fg_skip_li = self.fg_generator(latent, p_code, c_code, noise=noise, randomize_noise=randomize_noise)

        b_skip = bg_skip_li[-1]
        p_skip = fg_skip_li[self.p_depth]
        c_skip = fg_skip_li[-1]
        b_image = b_skip[:, 0:3, :, :]
        p_image = p_skip[:, 0:3, :, :]
        p_mask = p_skip[:, 3:4, :, :]
        p_mask = torch.sigmoid(p_mask)
        c_image = c_skip[:, 0:3, :, :]
        c_mask = c_skip[:, 3:4, :, :]
        c_mask = torch.sigmoid(c_mask)

        b_mkd = (torch.ones_like(c_mask)-c_mask) * b_image
        p_mkd = p_mask * p_image
        c_mkd = c_mask * c_image
        # fnl_image = b_mkd + c_mkd
        fnl_image = c_image

        raw_images = [b_image, p_image, c_image]
        # mkd_images = [b_mkd, p_mkd, c_mkd]
        mkd_images = raw_images
        masks = [p_mask, c_mask]

        if return_latents:
            return [fnl_image, raw_images, mkd_images, masks], [b_code, p_code, c_code], latent
        else:
            return [fnl_image, raw_images, mkd_images, masks], [b_code, p_code, c_code], None


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
    def __init__(self, size, pred_len, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
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
        self.rf_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

        self.pred_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], pred_len),
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
        rf_score = self.rf_linear(out)
        code_pred = self.pred_linear(out)

        return [rf_score, code_pred]


#################### bg discriminator #################
class MnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(
            1, 1, kernel_size, stride, padding, dilation, groups, False)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        """
        input is regular tensor with shape N*C*H*W
        mask has to have 1 channel N*1*H*W
        """
        output = self.input_conv(input)
        if mask is not None:
            mask = self.mask_conv(mask)
        return output, mask


class downBlock_mnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = MnetConv(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input, mask=None):
        """
        input is regular tensor with shape N*C*H*W
        mask has to have 1 channel N*1*H*W
        """
        output, mask = self.conv(input, mask)
        # output = self.bn(output)
        output = F.leaky_relu(output, 0.2, inplace=True)
        output = F.avg_pool2d(output, 2)
        if mask is not None:
            mask = F.avg_pool2d(mask, 2)
        return output, mask


def fromRGB_layer(out_planes):
    layer = nn.Sequential(
        nn.Conv2d(3, out_planes, 1, 1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return layer


class D_NET_BG_BASE(nn.Module):
    def __init__(self, ndf):
        super().__init__()
        self.df_dim = ndf
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        self.conv = MnetConv(ndf, ndf * 2, 3, 1, 1)
        self.conv_uncond_logits1 = MnetConv(ndf * 2, 1, 3, 1, 1)
        self.conv_uncond_logits2 = MnetConv(ndf * 2, 1, 3, 1, 1)

    def forward(self, x_code, mask):
        x_code, mask = self.conv(x_code, mask)
        x_code = F.leaky_relu(x_code, 0.2, inplace=True)
        _x_code, _mask = self.conv_uncond_logits1(x_code, mask)
        classi_score = torch.sigmoid(_x_code)
        _x_code, _mask = self.conv_uncond_logits2(x_code, mask)
        rf_score = torch.sigmoid(_x_code)
        return [classi_score, rf_score], _mask


class D_NET_BG(nn.Module):
    def __init__(self, start_depth):
        super().__init__()
        self.df_dim = 256
        self.start_depth = start_depth
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        self.from_RGB_net = nn.ModuleList([fromRGB_layer(ndf)])
        self.down_net = nn.ModuleList([D_NET_BG_BASE(ndf)])
        ndf = ndf // 2
        for _ in range(self.start_depth):
            self.from_RGB_net.append(fromRGB_layer(ndf))
            self.down_net.append(downBlock_mnet(ndf, ndf * 2, 3, 1, 1))
            ndf = ndf // 2

        self.df_dim = ndf
        self.cur_depth = self.start_depth

    def forward(self, x_var, alpha=None, mask=None):
        x_code = self.from_RGB_net[self.cur_depth](x_var)
        for i in range(self.cur_depth, -1, -1):
            x_code, mask = self.down_net[i](x_code, mask)
            if i == self.cur_depth and i != self.start_depth and alpha < 1:
                y_var = F.avg_pool2d(x_var, 2)
                y_code = self.from_RGB_net[i-1](y_var)
                x_code = (1 - alpha) * y_code + alpha * x_code

        classi_score = x_code[0]
        rf_score = x_code[1]
        return [rf_score, classi_score, mask]