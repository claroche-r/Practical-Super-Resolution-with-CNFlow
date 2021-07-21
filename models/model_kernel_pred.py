from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn.parallel import DataParallel  # , DistributedDataParallel

from models.select_network import define_G
from models.model_base import ModelBase
from models.network_sr import RealSRMD, RealSRMD_RRDB
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPrediction(ModelBase):
    def __init__(self, opt):
        super(ModelPrediction, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = define_G(opt).to(self.device)
        self.netG = DataParallel(self.netG)

        try:
            self.netEncoder = RealSRMD(in_nc=54, out_nc=3, nc=128, nb=12, upscale=4,
                                       act_mode='R', upsample_mode='pixelshuffle')

            self.netEncoder.load_state_dict(torch.load(opt['path_RealSRMD']), strict=True)
        except:
            self.netEncoder = RealSRMD_RRDB(in_nc=54, out_nc=3, gc= 32, nc=64, nb=8, upscale=4, act_mode='R',
                                            upsample_mode='upconv')

            self.netEncoder.load_state_dict(torch.load(opt['path_RealSRMD']), strict=True)

        self.netEncoder.eval()
        for k, v in self.netEncoder.named_parameters():
            v.requires_grad = False
        self.netEncoder.to(self.device)

        self.netEncoder = self.netEncoder.kernel_encoder


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
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
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_loss_type = self.opt_train['G_loss_type']
        if G_loss_type == 'l1':
            self.G_loss = nn.L1Loss().to(self.device)
        elif G_loss_type == 'l2':
            self.G_loss = nn.MSELoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_loss))

        self.G_loss_weight = self.opt_train['G_loss_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

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
        self.L = data['L'].to(self.device)
        self.k = data['kernel'].to(self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.k_pred = self.netG(self.L)
        self.k_true_encoded = self.netEncoder(self.k)
        G_loss = self.G_loss_weight * self.G_loss(self.k_pred, self.k_true_encoded)
        G_loss.backward()

        self.G_optimizer.step()

        self.log_dict['G_loss'] = G_loss.item()
 
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.k_pred = self.netG(self.L)
        self.netG.train()
        self.k_true_encoded = self.netEncoder(self.k)

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['k_pred'] = self.k_pred.detach().float().cpu()
        out_dict['k'] = self.k_pred.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
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
