from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.noise = None

    def forward(self, image, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(image.size(0), 1, image.size(2), image.size(3), 
                                device=image.device, dtype=image.dtype)
        elif noise is None:
            noise = self.noise

        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
        
    def forward(self, x, eps=1e-8):
        stddev = torch.sqrt(
            torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True) + eps)
        inject_shape = list(x.size())[:]
        inject_shape[1] = 1
        inject = torch.mean(stddev, dim=1, keepdim=True)
        inject = inject.expand(inject_shape)
        return torch.cat((x, inject), dim=1)

class Equalized_LR:
    def __init__(self, name):
        self.name = name
        
    def compute_weight(self, module):
        weight = getattr(module, self.name+'_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()        
        return weight * sqrt(2 / fan_in)
    
    @staticmethod
    def apply(module, name):
        fn = Equalized_LR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn
    
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)
        
def equal_lr(module, name='weight'):
    Equalized_LR.apply(module, name)
    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)
    
class EqualConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        deconv = nn.ConvTranspose2d(*args, **kwargs)
        deconv.weight.data.normal_()
        deconv.bias.data.zero_()
        self.deconv = equal_lr(deconv)
        
    def forward(self, input):
        return self.deconv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, mapping_linear=False, 
                 gain=None, lrmul=None, bias=True, use_wscale=False):
        super().__init__()

        self.mapping_linear = mapping_linear
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        if mapping_linear:
            he_std = gain * in_dim ** (-0.5)
            if use_wscale:
                init_std = 1.0 / lrmul
                self.w_mul = he_std * lrmul
            else:
                init_std = he_std / lrmul
                self.w_mul = lrmul
            self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim) * init_std)
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(out_dim))
                self.b_mul = lrmul
            else:
                self.bias = None
        else:
            self.linear = equal_lr(linear)

    def forward(self, input):
        if self.mapping_linear:
            bias = self.bias
            if bias is not None:
                bias = bias * self.b_mul
            return F.linear(input, self.weight * self.w_mul, bias)
        else:
            return self.linear(input)

class PixelNorm(nn.Module):
    """
    Pixel wize normalization
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    
class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out
    
class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out
    
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
    
class Blur(nn.Module):
    def __init__(self, kernel=None, normalize=None, flip=None, stride=1):
        super(Blur, self).__init__()
        self.normalize = normalize
        self.flip = flip
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = self.create_kernel(kernel)
        
        #if upsample_scale_factor > 1:
        #    kernel = kernel * (upsample_scale_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.stride = stride
        
        
    def create_kernel(self, kernel):
        kernel = torch.tensor(kernel).float().view(1, len(kernel))
        kernel = kernel.t() * kernel
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        if self.normalize:
            kernel /= kernel.sum()
        if self.flip:
            kernel = torch.flip(kernel, [2, 3])
        return kernel
    
    def forward(self, input):
        kernel = self.kernel.expand(input.size(1), -1, -1, -1)
        padding_size = int((self.kernel.size(2) - 1) / 2)
        return F.conv2d(input, kernel, stride=self.stride, 
                        padding=padding_size, groups=input.size(1))
    
    
class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super(Truncation, self).__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)
    

