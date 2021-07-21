import math
import torch.nn as nn
import models.basicblock as B
import torch

"""
# --------------------------------------------
# RealSRMD (15 conv layers)
# --------------------------------------------
Reference:
@inproceedings{zhang2018learning,
  title={Learning a single convolutional super-resolution network for multiple degradations},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3262--3271},
  year={2018}
}
http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Learning_a_Single_CVPR_2018_paper.pdf
"""


class RRDB(nn.Module):
    """
    gc: number of growth channels
    nb: number of RRDB
    """
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=23, gc=32, bias=True, upscale=4,
                 act_mode='L', upsample_mode='upconv', blur_kernel_size = 33):
        super(RRDB, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.kernel_features_size = 50
        self.hidden_size_kernel = 50

        m_head = B.conv(in_nc, nc, mode='C', bias=bias)

        m_body = [B.RRDB(nc, gc=32, mode='C'+act_mode, bias=bias) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C', bias=bias))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode, bias=bias)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode, bias=bias) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode, bias=bias)
        H_conv1 = B.conv(nc, out_nc, mode='C', bias=bias)
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)

        self.kernel_encoder = B.sequential(nn.Linear(blur_kernel_size * blur_kernel_size, self.hidden_size_kernel),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size_kernel, self.kernel_features_size))

    def forward(self, x, kernel, iso):
        B, _, H, W = x.size()
        kernel_features = self.kernel_encoder(kernel).view(B, self.kernel_features_size, 1, 1)
        kernel_features = kernel_features.repeat(1, 1, H, W)
        iso_map = iso.view(B, 1, 1, 1).repeat(1, 1, H, W)

        x_extented = torch.cat((x, iso_map, kernel_features), 1)
        res = self.model(x_extented)
        return res


if __name__ == '__main__':
    from utils import utils_model
    #model = SRMD(in_nc=18, out_nc=3, nc=64, nb=15, upscale=4, act_mode='R', upsample_mode='pixelshuffle')
    #print(utils_model.describe_model(model))

    #x = torch.randn((2, 3, 100, 100))
    #k_pca = torch.randn(2, 15, 1, 1)
    #x = model(x, k_pca)
    #print(x.shape)

    #  run models/network_srmd.py
