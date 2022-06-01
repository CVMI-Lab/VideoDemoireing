
import os
from config.config import args
os.environ['CUDA_VISIBLE_DEVICES'] = "%s" % args.GPU_ID
import numpy as np
import torch
import argparse
import cv2
import scipy.io as io
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
from model.VDM_PCD import VDM_PCD, model_fn_decorator
from data.load_video_temporal import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
import lpips


def test(args, TestImgLoader, model, model_fn_test, net_metric, epoch):

    save_path = args.TEST_RESULT_DIR + '/' + '%04d' % epoch
    mkdir(save_path)
    tbar = tqdm(TestImgLoader)

    if args.have_gt:
        f = open(args.TEST_RESULT_DIR + '/%04d_' % epoch + 'result.txt', "w")
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        for batch_idx, data in enumerate(tbar):
            with torch.no_grad():
                model.eval()
                loss, cur_psnr, cur_ssim, cur_lpips = model_fn_test(args, data, model, net_metric, save_path)
                number = data['number']
                f.write('%06s: LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f \n' % (number[0], cur_lpips, cur_psnr, cur_ssim))

                # metrics
                total_loss += loss.item()
                avg_val_loss = total_loss / (batch_idx+1)
                total_psnr += cur_psnr
                avg_val_psnr = total_psnr / (batch_idx+1)
                total_ssim += cur_ssim
                avg_val_ssim = total_ssim / (batch_idx+1)
                total_lpips += cur_lpips
                avg_val_lpips = total_lpips / (batch_idx+1)
                desc = 'Test: Epoch %d, Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f, Avg. Loss = %.5f' % (
                    epoch, avg_val_lpips, avg_val_psnr, avg_val_ssim, avg_val_loss)
                tbar.set_description(desc)
                tbar.update()
        f.write('Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f \n' % (avg_val_lpips, avg_val_psnr, avg_val_ssim))
        f.close()
    else:
        for batch_idx, data in enumerate(tbar):
            with torch.no_grad():
                model.eval()
                model_fn_test(args, data, model, save_path)


def init():
    # Make dirs
    mkdir(args.TEST_RESULT_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID
    logger = SummaryWriter(args.LOGS_DIR)

    # initialize lpips
    net_metric_alex = lpips.LPIPS(net='alex').cuda()

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    return net_metric_alex


def load_checkpoint(model, load_epoch):
    load_dir = args.MODEL_DIR + '/nets/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    avg_lpips = torch.load(load_dir)['avg_val_lpips']
    avg_psnr = torch.load(load_dir)['avg_val_psnr']
    avg_ssim = torch.load(load_dir)['avg_val_ssim']
    print('Avg. LPIPS, PSNR and SSIM values recorded from the checkpoint: %f, %f, %f' % (avg_lpips, avg_psnr, avg_ssim))
    model_state_dict = torch.load(load_dir)['state_dict']
    model.load_state_dict(model_state_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters


if __name__ == '__main__':
    # Create test list
    test_input_list = sorted([file for file in os.listdir(args.TEST_DATASET + '/target') if (file.endswith('.jpg') or file.endswith('.png'))])

    net_metric = init()
    model = VDM_PCD(args).cuda()
    learning_rate, iters = load_checkpoint(model, args.TEST_EPOCH)
    loss_fn = multi_VGGPerceptualLoss().cuda()

    # create mode, data loader
    if args.have_gt:
        model_fn_test = model_fn_decorator(loss_fn=loss_fn, mode='val')
        TestImgLoader = data.DataLoader(data_loader(args, test_input_list, mode='val'),
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=0)
    else:
        model_fn_test = model_fn_decorator(loss_fn=loss_fn, mode='test')
        TestImgLoader = data.DataLoader(data_loader(args, test_input_list, mode='test'),
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=0)

    # test
    test(args, TestImgLoader, model, model_fn_test, net_metric, args.TEST_EPOCH)
    

