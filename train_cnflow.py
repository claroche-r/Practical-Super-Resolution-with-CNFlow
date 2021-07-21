import os.path
import math
import argparse
import random
import numpy as np
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main():

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    current_step = opt['current_step'] if opt['current_step'] else 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure tensorboard
    # ----------------------------------------
    writer = SummaryWriter(os.path.join(opt['path']['log'], 'tensorboard'))

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    model.init_train()
    model.load()
    logger.info(model.info_network())
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    n_epochs = opt["train"]["n_epochs"] if opt["train"]["n_epochs"] else 200

    for epoch in range(n_epochs):
        with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                        os.path.join(opt['path']['log'], 'tensorboard'))) \
                as profiler:
            for i, train_data in enumerate(train_loader):

                current_step += 1

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                model.update_learning_rate(current_step)

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                model.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                model.optimize_parameters(current_step)

                #profiler.step()

                # -------------------------------
                # 4) training information
                # -------------------------------
                if current_step % opt['train']['checkpoint_print'] == 0:
                    logs = model.current_log()  # such as loss
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                    for k, v in logs.items():  # merge log information into message
                        message += '{:s}: {:.3e} '.format(k, v)
                        writer.add_scalar(k, v, current_step)
                    logger.info(message)


                # -------------------------------
                # 5) save model
                # -------------------------------
                if current_step % opt['train']['checkpoint_save'] == 0:
                    logger.info('Saving the model.')
                    model.save(current_step)

                # -------------------------------
                # 6) testing
                # -------------------------------
                if current_step % opt['train']['checkpoint_test'] == 0:

                    avg_likelihood = 0.0
                    idx = 0

                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['clean_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)

                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(test_data)
                        nll = model.test(1.0)

                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals['E'])
                        clean_img = util.tensor2uint(visuals['clean'])

                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        if current_step % opt['train']['checkpoint_save'] == 0:
                            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                            util.imsave(E_img, save_img_path)

                        logger.info('{:->4d}--> {:>10s} | nll: {:<4.2f}'.format(idx, image_name_ext, nll))

                        avg_likelihood += nll

                    avg_likelihood = avg_likelihood / idx

                    writer.add_images('Test set generation', np.concatenate((clean_img, E_img), axis=0), epoch,
                                      dataformats='HWC')
                    writer.add_scalar('PSNR', avg_likelihood, current_step)

                    # testing log
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average likelihood : {:<.2f}dB\n'.format(epoch,
                                                                                                     current_step,
                                                                                                     avg_likelihood))
    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
