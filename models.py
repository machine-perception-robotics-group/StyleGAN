import numpy as np
import torch
import torch.nn as nn

from Custom_layers import *
    
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride,
                 padding, down=False, fused=False, d_last=False):
        super(ConvBlock, self).__init__()
        
        if d_last:
            self.stddev = MinibatchStddev()
            in_features = in_features+1
        else:
            self.stddev = None
        
        conv_layer1 = [
            EqualConv2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        
        if d_last:
            kernel_size = 4
            stride = 1
            padding = 0
            
        if down:
            if fused:
                conv_layer2 = [
                    Blur([1, 2, 1], True, True, 1),
                    FusedDownsample(out_features, out_features, kernel_size=kernel_size, padding=padding),
                    nn.LeakyReLU(negative_slope=0.2)
                ]
            else:
                conv_layer2 = [
                    Blur([1, 2, 1], True, True, 1),
                    EqualConv2d(out_features, out_features, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=True),
                    nn.AvgPool2d(2, 2),
                    nn.LeakyReLU(negative_slope=0.2)
                ]
        else:
            conv_layer2 = [
                EqualConv2d(out_features, out_features, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=True),
                nn.LeakyReLU(negative_slope=0.2)
            ]
        
        self.c1 = nn.Sequential(*conv_layer1)
        self.c2 = nn.Sequential(*conv_layer2)
    
    def forward(self, x):
        if self.stddev:
            x = self.stddev(x)
        h = self.c1(x)
        out = self.c2(h)
        return out
    
class GBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding,
                 up=False, fused=False, n_style=512, initial=False):
        super(GBlock, self).__init__()
        if not initial:
            if up:
                if fused:
                    self.c1 = nn.Sequential(
                        FusedUpsample(in_features, out_features, kernel_size=kernel_size, padding=padding),
                        Blur([1, 2, 1], True, True, 1))
                else:
                    self.c1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=True),
                        Blur([1, 2, 1], True, True, 1)
                    )
            else:
                self.c1 = EqualConv2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=True)
                
        self.initial = initial
        self.make_noise1 = NoiseInjection(out_features)
        self.adain1 = AdaptiveInstanceNorm(out_features, n_style)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.c2 = nn.Sequential(
            EqualConv2d(out_features, out_features, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=True))
        self.make_noise2 = NoiseInjection(out_features)
        self.adain2 = AdaptiveInstanceNorm(out_features, n_style)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x, style_code):
        out = x
        if not self.initial:
            out = self.c1(out)
        out = self.make_noise1(out, noise=None)
        out = self.lrelu1(out)
        out = self.adain1(out, style_code[:,0])

        out = self.c2(out)
        out = self.make_noise2(out, noise=None)
        out = self.lrelu2(out)
        out = self.adain2(out, style_code[:,1])
        return out
    
class StyleGenerator(nn.Module):
    def __init__(self, switch_ch=4, n_block=9, n_ch=512, n_style=512):
        super(StyleGenerator, self).__init__()
        module_list = []
        conv_module_list = []
        fused = False
        in_ch = n_ch
        out_ch = n_ch
        for block_idx in range(1, n_block+1):
            if block_idx == 1:initial = True
            else:initial = False
            module_list.append(GBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, up=True, fused=fused, initial=initial))
            conv_module_list.append(ConvBlock(out_ch, 3, kernel_size=1, stride=1, padding=0))
            if block_idx >= switch_ch:
                fused = True
                in_ch = out_ch
                out_ch = in_ch // 2
            
        self.const = ConstantInput(n_ch)
        self.main_net = nn.ModuleList(module_list)
        self.to_rgb = nn.ModuleList(conv_module_list)
        
    def forward(self, style_code, depth=0, alpha=1):
        const = self.const(style_code.size(0))
        
        if depth > 0 and alpha < 1.0:
            h = const
            h = self.main_net[0](h, style_code[:, 0:2])
            for i in range(1, depth - 1):
                h = self.main_net[i](h, style_code[:, (i*2):(i*2)+2])
            
            h1 = self.main_net[depth - 1](h, style_code[:, ((depth-1)*2):((depth-1)*2)+2])
            h2 = F.upsample(h1, scale_factor=2, mode='nearest')
            h3 = self.to_rgb[depth - 1](h2)
            h4 = self.main_net[depth](h1, style_code[:, ((depth)*2):((depth)*2)+2])
            h4 = self.to_rgb[depth](h4)
            
            h = h3 - alpha * (h3 - h4)
        else:
            h = const
            h = self.main_net[0](h, style_code[:, 0:2])
            for i in range(1, depth+1):
                h = self.main_net[i](h, style_code[:, (i*2):(i*2)+2])
            
            h = self.to_rgb[depth](h)
        
        return h    
        
class Generator(nn.Module):
    def __init__(self, switch_ch=4, img_size=1024, n_ch=512, n_style=512, n_mapping_layers=8,
                 truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995, style_mixing_prob=0.9):
        super(Generator, self).__init__()
        n_block = int(np.log2(img_size))
        
        mapping_network = [PixelNorm()]
        for i in range(n_mapping_layers):
            mapping_network.append(EqualLinear(n_style, n_style, mapping_linear=True, 
                                   gain=2 ** 0.5, lrmul=0.01, bias=True, use_wscale=True))
            mapping_network.append(nn.LeakyReLU(negative_slope=0.2))
        
        self.mapping_network = nn.Sequential(*mapping_network)
        self.style_generator = StyleGenerator(switch_ch=switch_ch, 
                                              n_block=n_block, 
                                              n_ch=n_ch, 
                                              n_style=n_style)
        
        self.style_mixing_prob = style_mixing_prob
        self.n_layers = (n_block - 1) * 2
        if truncation_psi > 0:
            self.truncation = Truncation(torch.zeros(n_style),
                                         truncation_cutoff,
                                         truncation_psi,
                                         dlatent_avg_beta)
        else:
            self.truncation = None
        
        self.n_AdaIN = 0
        for m in self.style_generator.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm':    self.n_AdaIN += 1
        
    def forward(self, style, depth=0, alpha=-1, true_size=1024, training=True):
        style_code = self.mapping_network(style).unsqueeze(1).expand(-1, self.n_AdaIN, -1)
        
        if training:
            if self.truncation is not None:
                self.truncation.update(style_code[0, 0].detach())
            
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                style2 = torch.randn(style.shape).to(style.device)
                
                style_code2 = self.mapping_network(style2).unsqueeze(1).expand(-1, self.n_AdaIN, -1)
                layer_idx = torch.arange(self.n_AdaIN)[None,:,None].to(style.device)
                #cur_layers = 2 * (depth + 1)
            
                mixing_cutoff = np.random.randint(1, self.n_AdaIN) if np.random.random() < self.style_mixing_prob else self.n_AdaIN
                style_code = torch.where(layer_idx < mixing_cutoff, style_code, style_code2)
        
            if self.truncation is not None:    style_code = self.truncation(style_code)
        
        return self.style_generator(style_code, depth, alpha)
        
class Discriminator(nn.Module):
    def __init__(self, switch_ch=4, img_size=1024, n_ch=16):
        super(Discriminator, self).__init__()
        n_block = int(np.log2(img_size))
        self.total_depth = n_block - 1
        
        module_list = []
        conv_module_list = []
        down = True
        fused = False
        d_last = False
        in_ch = n_ch
        out_ch = n_ch
        for block_idx in range(1, n_block+1):
            if block_idx == n_block:
                down = False
                d_last = True
            conv_module_list.append(nn.Sequential(EqualConv2d(3, in_ch, kernel_size=1, stride=1, padding=0),
                                    nn.LeakyReLU(negative_slope=0.2)))
            module_list.append(ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, down=down, fused=fused, d_last=d_last))
            if block_idx <= switch_ch + 1:
                fused = True
                in_ch = out_ch
                out_ch = in_ch * 2
            else:
                in_ch = out_ch
            
        self.main_net = nn.Sequential(*module_list)
        self.from_rgb = nn.Sequential(*conv_module_list)
        self.linear = EqualLinear(out_ch, 1)
        
    def forward(self, x, depth=0, alpha=0):
        if depth > 0 and alpha < 1.0:
            h1 = self.from_rgb[self.total_depth - depth](x)
            h1 = self.main_net[self.total_depth - depth](h1)
            x2 = F.avg_pool2d(x, 2, 2)
            h2 = F.leaky_relu(self.from_rgb[self.total_depth - depth+1](x2))
            h = h2 - alpha * (h2 - h1)
            
        else:
            h = self.from_rgb[self.total_depth - depth](x)
            h = self.main_net[self.total_depth - depth](h)
            
        for i in range(depth):
            h = self.main_net[self.total_depth - depth+1+i](h)
            
        out = self.linear(h.view(h.size(0), -1))
        return out