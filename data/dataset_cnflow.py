import random
import torch.utils.data as data
import utils.utils_image as util
import os
import torch


class DatasetCNFlow(data.Dataset):
    """
        Folder must be as follows:
                    -dataroot |-clean|-aaa.png
                                     |-bbb.png
                                     |-...
                              |-100|-aaa.png
                                     |-bbb.png
                                     |-...
                              |-...|-aaa.png
                                     |-bbb.png
                                     |-...
                              |-3200 |-aaa.png
                                     |-bbb.png
                                     |-...
    """
    def __init__(self, opt):
        super(DatasetCNFlow, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 128
        self.ISO_list = self.opt['ISO_list'] if self.opt['ISO_list'] else ['400', '800', '1600', '3200']

        print("Different ISO in the training data:", self.ISO_list)
        self.paths_noisy = {}
        self.paths_clean = util.get_image_paths(os.path.join(opt['dataroot'], 'clean'))
        for iso in self.ISO_list:
            self.paths_noisy[iso] = util.get_image_paths(os.path.join(opt['dataroot'], iso))

        # Add the clean image as ISO 0. ISO 0 correspond here to no noise.
        self.ISO_list.append('0')
        self.paths_noisy['0'] = self.paths_clean

    def __getitem__(self, index):
        iso = random.choice(self.ISO_list)

        # Get noisy image
        noisy_path = self.paths_noisy[iso][index]
        img_noisy = util.imread_uint(noisy_path, self.n_channels)
        img_noisy = util.uint2single(img_noisy) - 0.5  # rescaling between [-0.5,0.5] helps consistency

        # Get clean image
        clean_path = self.paths_clean[index]
        img_clean = util.imread_uint(clean_path, self.n_channels)
        img_clean = util.uint2single(img_clean) - 0.5  # rescaling between [-0.5,0.5] helps consistency

        if self.opt['phase'] == 'train':
            H, W, C = img_clean.shape

            # randomly crop the clean/noisy patches
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_clean = img_clean[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_noisy = img_noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation - flip and/or rotate
            mode = random.randint(0, 7)
            img_clean, img_noisy = util.augment_img(img_clean, mode=mode), util.augment_img(img_noisy, mode=mode)

        # np to torch
        img_clean, img_noisy = util.single2tensor3(img_clean), util.single2tensor3(img_noisy)

        iso_value = int(iso)
        iso_map = torch.FloatTensor([iso_value/6400]).repeat(1, img_clean.size()[-2], img_clean.size()[-1])
        img_clean = torch.cat((img_clean, iso_map), 0)

        return {'clean': img_clean, 'noisy': img_noisy, 'noisy_path': noisy_path, 'clean_path': clean_path}

    def __len__(self):
        return len(self.paths_clean)
