import random
import numpy as np

import torch
import torch.utils.data as data

import utils.utils_image as util
import utils.utils_cnflow as utils_cnflow
from utils import utils_sisr as sisr

from scipy.io import loadmat
from scipy import ndimage

from imgaug.augmenters.arithmetic import compress_jpeg


class DatasetSR(data.Dataset):
    ''' Sythesize with custom blur kernel, downsampling kernel and noise on-the-fly '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 256
        self.L_size = self.patch_size // self.sf

        self.blur_kern = loadmat(opt['custom_blur_kern_path'])['kernels'][0]
        self.down_kern = loadmat(opt['custom_down_kern_path'])['kernels']

        # CNFlow opt
        self.ISO_list = [0., 400., 800., 1600.]
        self.opt_cnflow = {"in_nc": 3, "out_nc": 3, "n_flow": 8, "affine_nc": 64,
                           "nb": 3, "in_nc_cond": 4, "LU_decomposed": True,"actnorm_scale": 1.,
                           "affine_eps": 0.0001, "init_type": "orthogonal", "init_bn_type": "normal",
                           "init_gain": 0.2}

        cnflow_path = opt['cnflow-path']
        print(type(cnflow_path))
        self.load_cnflow(cnflow_path)

        # ------------------------------------
        # get paths of H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot'])

        assert self.paths_H, 'Error: H path is empty.'

    def load_cnflow(self, path):
        from models.module_cnflow import CNFlow as net
        self.cnflow = net(3, 4, n_blocks=3, actnorm_scale=1, LU_decomposed=True, opt=self.opt_cnflow)
        self.cnflow.load_state_dict(torch.load(path), strict=True)
        self.cnflow.eval()
        for k, v in self.cnflow.named_parameters():
            v.requires_grad = False
        self.cnflow.to('cpu')

    def add_noise(self, img, iso, heat=1):
        img = img - 0.5   # CNFlow takes input between [-0.5,0.5]
        z = utils_cnflow.get_z(heat, img, nb=3).to('cpu')
        iso = iso.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        iso_map = iso.repeat(1, 1, img.size()[-2], img.size()[-1])
        img_extented = torch.cat([img, iso_map], axis=1).to('cpu')
        noisy, _ = self.cnflow(cond=img_extented, z=z, reverse=True)
        noisy = noisy + 0.5
        return noisy.to('cpu')

    def GaussianBlur(self, img):
        l_max = 10
        theta = np.pi * random.random()
        l1 = 0.1 + l_max * random.random()
        l2 = 0.1 + (l1 - 0.1) * random.random()

        kernel = sisr.anisotropic_Gaussian(ksize=33, theta=theta, l1=l1, l2=l2)

        out = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='wrap')
        return out, kernel

    def CustomBlur(self, img):
        index_blur = random.randint(0, len(self.blur_kern) - 1)
        kernel = sisr.extend_kernel_size(self.blur_kern[index_blur], (33, 33))
        out = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='wrap')
        return out, kernel

    def KernelGANDownsampling(self, img):
        index_down = random.randint(0, len(self.down_kern) - 1)
        kernel = self.down_kern[index_down, :, :]
        kernel /= kernel.sum()
        out = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='wrap')
        out = out[::self.sf, ::self.sf]
        return out, kernel

    def BicubicDownsampling(self, img):
        kernel_np = sisr.build_bicubic_filter(self.sf)
        kernel_np = sisr.extend_kernel_size(kernel_np, size=(33, 33))
        out = sisr.bicubic_degradation(img, sf=self.sf)
        return out, kernel_np

    def __getitem__(self, index):
        # get H image
        H_path = self.paths_H[index]
        img_H = util.uint2single(util.imread_uint(H_path, self.n_channels))

        # modcrop
        img_H = util.modcrop(img_H, self.sf)

        # randomly crop the H image
        H, W, C = img_H.shape
        crop_size = self.L_size * self.sf + 80  # We use padding to avoid border effects
        rnd_h = random.randint(0, max(0, H - crop_size))
        rnd_w = random.randint(0, max(0, W - crop_size))
        img_H = img_H[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]

        # data augmentation
        mode = random.randint(0, 7)
        img_H = util.augment_img(img_H, mode=mode).copy()

        # Blurring
        blur_method = random.choice(['motion', 'gaussian', 'no_blur'])

        if blur_method == 'motion':
            blurred, blur_ker = self.CustomBlur(img_H)

        elif blur_method == 'gaussian':
            blurred, blur_ker = self.GaussianBlur(img_H)

        else:
            blurred = img_H
            blur_ker = np.zeros((33, 33))
            blur_ker[17, 17] = 1

        # Downsampling
        down_method = random.choice(['kernelgan', 'bicubic'])

        if down_method == 'kernelgan':
            img_L, down_ker = self.KernelGANDownsampling(blurred)

        else:
            img_L, down_ker = self.BicubicDownsampling(blurred)

        # Remove padding
        img_H = img_H[40:-40, 40:-40, :]
        img_L = img_L[40 // self.sf:-(40 // self.sf), 40 // self.sf:-(40 // self.sf), :]

        # Getting resulting kernel
        ker = ndimage.filters.convolve(np.repeat(np.expand_dims(blur_ker, axis=2), 3, 2),
                                       np.expand_dims(down_ker, axis=2),
                                       mode='wrap')[:, :, 0]

        # np to tensor
        img_H = util.single2tensor4(img_H)
        img_L = util.single2tensor4(img_L)

        ker = np.reshape(ker, (-1), order="F")
        ker = ker / ker.sum()
        ker = torch.FloatTensor(ker)

        # Add noise
        iso = random.choice(self.ISO_list)

        # Loop that avoid CNFlow crashes
        if iso == 0.:
            iso = torch.FloatTensor([0.])
            noisy = img_L

        else:
            iso = torch.FloatTensor([iso / 6400.])
            heat = 0.7
            bool_ = True

            while bool_:
                noisy = self.add_noise(img_L, iso, heat=heat)

                if -2 < noisy.min() and noisy.max() < 2:
                    bool_ = False

                elif heat <= 0.6:
                    iso = torch.FloatTensor([0.])
                    noisy = img_L
                    bool_ = False

                else:
                    heat = heat - 0.1

        img_L = noisy
        img_L.clamp(0,1)
        img_L = img_L[0]
        img_H = img_H[0]
        L_path = H_path

        # JPEG compression
        qf = random.choice([50, 60, 70, 80, 90, 100])

        if not qf == 100:
            img_L = util.tensor2uint(img_L)
            img_L = compress_jpeg(img_L, qf)
            img_L = util.uint2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'kernel': ker, 'iso': iso, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
