import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import models.basicblock as B


class VGG_40(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, out_nc=1, ac_type='BL'):
        super(VGG_40, self).__init__()
        # features
        # hxw, c
        # 40, 3
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 20, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 10, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 5, 256
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 100), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(100, out_nc))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, out_nc=1, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGAN, self).__init__()

        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, out_nc, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64, out_nc=50, use_bias=True):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, out_nc, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view(flat.size()[:2]) # torch size: [B, code_len]
