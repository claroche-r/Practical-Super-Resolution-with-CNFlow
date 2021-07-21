import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.module_cnflow_utils as utils
from math import log, pi


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", do_actnorm=True,
                 weight_std=0.05):

        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))

        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()

        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)

        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", logscale_factor=3):

        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))

        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class InvertibleConv1x1(nn.Module):

    def __init__(self, num_channels, LU_decomposed=False):

        super().__init__()

        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = utils.pixels(input)
        dlogdet = torch.slogdet(self.weight)[1] * pixels

        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)

        else:
            weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)

        return weight, dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """ log-det = log|abs(|W|)| * pixels """

        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)

            if logdet is not None:
                logdet = logdet + dlogdet

            return z, logdet

        else:
            z = F.conv2d(input, weight)

            if logdet is not None:
                logdet = logdet - dlogdet

            return z, logdet


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """ lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
            Var = logs ** 2
        """

        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)

        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)

        return utils.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(logs) * eps_std)

        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        eps = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape) * eps_std)
        return eps


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return

        if (self.bias != 0).any():
            self.inited = True
            return

        assert input.device == self.bias.device, (input.device, self.bias.device)

        with torch.no_grad():
            bias = utils.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = utils.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not reverse:
            return input + bias

        else:
            return input - bias

    def _scale(self, input, logdet=None, reverse=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not reverse:
            input = input * torch.exp(logs) # should have shape batchsize, n_channels, 1, 1

        else:
            input = input * torch.exp(-logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = utils.sum(logs) * utils.pixels(input)
            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False, offset_mask=None, logs_offset=None, bias_offset=None):
        if not self.inited:
            self.initialize_parameters(input)

        self._check_input_dim(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask

        # no need to permute dims as old version
        if not reverse:
            # center and scale

            # self.input = input
            input = self._center(input, reverse, bias_offset)
            input, logdet = self._scale(input, logdet, reverse, logs_offset)

        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse, logs_offset)
            input = self._center(input, reverse, bias_offset)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(self.num_features, input.size()))


class MaskedActNorm2d(ActNorm2d):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def forward(self, input, mask, logdet=None, reverse=False):

        assert mask.dtype == torch.bool
        output, logdet_out = super().forward(input, logdet, reverse)

        input[mask] = output[mask]
        logdet[mask] = logdet_out[mask]

        return input, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = utils.squeeze2d(input, self.factor)  # Squeeze in forward
            return output, logdet

        else:
            output = utils.unsqueeze2d(input, self.factor)
            return output, logdet


class AffineInjectorAndCoupling(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.opt_net = opt
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_cond = self.opt_net["in_nc_cond"]
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.hidden_channels = 64 if not self.opt_net['affine_nc'] else self.opt_net['affine_nc']
        self.affine_eps = self.opt_net['affine_eps']

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_cond,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_cond,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, cond=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Affine injector
            scaleFt, shiftFt = self.feature_extract(cond, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Affine coupling
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, cond, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = utils.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)

            scale, shift = self.feature_extract_aff(z1, cond, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = utils.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(cond, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return utils.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = utils.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = utils.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class FlowStep(nn.Module):
    def __init__(self, in_channels, actnorm_scale=1.0, LU_decomposed=False, opt=None):

        super().__init__()

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        self.affine = AffineInjectorAndCoupling(in_channels=in_channels, opt=opt)

    def forward(self, input, logdet=None, reverse=False, cond=None):
        if not reverse:
            return self.normal_flow(input, logdet, cond=cond)
        else:
            return self.reverse_flow(input, logdet, cond=cond)

    def normal_flow(self, z, logdet, cond=None):
        # 1. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.invconv(z, logdet=logdet, reverse=False)

        # 3. coupling
        z, logdet = self.affine(input=z, logdet=logdet, reverse=False, cond=cond)

        return z, logdet

    def reverse_flow(self, z, logdet, cond=None):
        # 1.coupling
        z, logdet = self.affine(input=z, logdet=logdet, reverse=True, cond=cond)

        # 2. permute
        z, logdet = self.invconv(z, logdet=logdet, reverse=True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class Block(nn.Module):
    @staticmethod
    def gaussian_log_p(x, mean, log_sd):
        return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

    @staticmethod
    def gaussian_sample(eps, mean, log_sd):
        return mean + torch.exp(log_sd) * eps

    def __init__(self, in_channel, n_flow, split=True, actnorm_scale=1.0, LU_decomposed=True, opt=None):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(FlowStep(squeeze_dim, actnorm_scale=actnorm_scale, LU_decomposed=LU_decomposed, opt=opt))

        self.split = split

        if split:
            self.prior = Conv2dZeros(in_channel * 2, in_channel * 4)

        else:
            self.prior = Conv2dZeros(in_channel * 4, in_channel * 8)

    def forward(self, input, cond=None, logdet=None, reverse=True, eps=None):
        if reverse:
            return self.reverse_flow(input, cond, logdet, eps=eps)

        else:
            return self.normal_flow(input, cond, logdet)

    def normal_flow(self, input, cond=None, logdet=None):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        for flow in self.flows:
            out, logdet = flow(out, logdet=logdet, reverse=False, cond=cond)
        """
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = self.gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = self.gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        """

        return out, logdet #, log_p, z_new

    def reverse_flow(self, output, cond=None, logdet=None, eps=None, reconstruct=False):
        input = output
        """
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = self.gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = self.gaussian_sample(eps, mean, log_sd)
                input = z
        """
        for flow in self.flows[::-1]:
            input, logdet = flow(input, logdet=logdet, reverse=True, cond=cond)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 4, height * 2, width * 2)

        return unsqueezed, logdet


class CNFlow(nn.Module):
    def __init__(self, in_nc, n_flow, n_blocks=4, actnorm_scale=1.0, LU_decomposed=False, opt=None):

        super().__init__()

        self.layers = nn.ModuleList()
        self.opt = opt
        n_channels = in_nc

        self.preFlow = FlowStep(in_nc, actnorm_scale=actnorm_scale, LU_decomposed=LU_decomposed, opt=opt)

        for _ in range(n_blocks-1):
            self.layers.append(Block(n_channels, n_flow, split=True, actnorm_scale=actnorm_scale,
                                     LU_decomposed=LU_decomposed, opt=opt))
            n_channels *= 4

        self.layers.append(Block(n_channels, n_flow, split=False, actnorm_scale=actnorm_scale,
                                 LU_decomposed=LU_decomposed, opt=opt))

    @staticmethod
    def downsample(img, scale_factor):
        n, p = img.shape[-2], img.shape[-1]
        downsampler = nn.AdaptiveAvgPool2d((n//scale_factor, p//scale_factor))
        return downsampler(img)

    def forward(self, gt=None, cond=None, z=None, logdet=0., reverse=False):

        if reverse:
            assert cond is not None
            assert z is not None
            out, logdet = self.decode(cond, z, logdet=logdet)
            out = out + cond[:,:3,:,:]

            return out, logdet

        else:
            assert gt is not None
            assert cond is not None
            z, logdet = self.encode(gt - cond[:,:3,:,:], cond, logdet=logdet)

            return z, logdet

    def encode(self, gt, cond, logdet=0.0):

        reverse = False
        fl_fea, logdet = self.preFlow(gt, cond=cond, logdet=logdet, reverse=reverse)

        for i,layer in enumerate(self.layers):
            fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=reverse,
                                   cond=self.downsample(cond, scale_factor=2**(i + 1)))

        z = fl_fea

        return z, logdet

    def decode(self, cond, z, logdet=0.0):
        fl_fea = z
        reverse = True
        n_blocks = len(self.layers)
        for i,layer in enumerate(reversed(self.layers)):
            fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=reverse, cond=self.downsample(cond, 2**(n_blocks - i)))

        fl_fea, logdet = self.preFlow(fl_fea, cond=cond, logdet=logdet, reverse=reverse)
        out = fl_fea

        return out, logdet
