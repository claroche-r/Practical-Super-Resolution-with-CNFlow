from collections import OrderedDict
import torch
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn.parallel import DataParallel  # DistributedDataParallel
import numpy as np

from models.select_network import define_G
from models.model_base import ModelBase
from models.module_cnflow import GaussianDiag


class CNFlowModel(ModelBase):
    def __init__(self, opt):
        super(CNFlowModel, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)

    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.opt['train']['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.clean = data['clean'].to(self.device)
        self.noisy = data['noisy'].to(self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        self.G_optimizer.zero_grad()

        z, logdet = self.netG(gt=self.noisy, cond=self.clean, reverse=False)

        objective = logdet
        objective = objective + GaussianDiag.logp(None, None, z)

        pixels = self.noisy.size()[-1] * self.noisy.size()[-2]
        nll = (-objective) / float(np.log(2.) * pixels)
        nll_loss = torch.mean(nll)
        loss_G_total = nll_loss * self.G_lossfn_weight

        loss_G_total.backward()
        self.G_optimizer.step()
        mean = loss_G_total.item()
        self.log_dict['G_loss'] = mean

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self, heat):
        self.netG.eval()
        self.fake_H = {}

        z = self.get_z(heat)

        with torch.no_grad():
            self.E, logdet = self.netG(cond=self.clean, z=z, reverse=True)

        with torch.no_grad():
            _, logdet = self.netG(gt=self.noisy, cond=self.clean, reverse=False)

        self.netG.train()

        objective = logdet
        objective = objective + GaussianDiag.logp(None, None, z).to(self.device)

        pixels = self.noisy.size()[-1] * self.noisy.size()[-2] * 3
        nll = (-objective) / float(np.log(2.) * pixels)
        nll_loss = torch.mean(nll)

        return nll_loss.item()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['clean'] = self.clean[:,:3,:,:].detach()[0].float().cpu() + 0.5
        out_dict['E'] = self.E.detach()[0].float().cpu() + 0.5
        return out_dict

    """
    # ----------------------------------------
    # Information of netG, netD and netF
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    def get_encode_nll(self, clean, noisy):
        self.netG.eval()
        with torch.no_grad():
            _, nll = self.netG(gt=noisy, cond=clean, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_noisy(self, clean, heat=None, z=None):
        return self.get_sr_with_z(clean, heat, z)[0]

    def get_encode_z(self, clean, noisy):
        self.netG.eval()
        with torch.no_grad():
            z, _ = self.netG(gt=noisy, cond=clean, reverse=False)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, noisy, clean):
        self.netG.eval()
        with torch.no_grad():
            z, nll = self.netG(gt=noisy, cond=clean, reverse=False)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, clean, heat=None):
        self.netG.eval()

        z = self.get_z(heat)

        with torch.no_grad():
            sr, logdet = self.netG(cond=clean, z=z, reverse=True)
        self.netG.train()
        return sr, z

    def get_z(self, heat):
        B, _, H, W = self.clean.size()
        C = self.opt['netG']['in_nc']
        factor = 2 ** self.opt['netG']['nb']
        H, W = H // factor, W // factor
        C = C * factor * factor
        size = (B, C, H, W)
        z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
        return z
