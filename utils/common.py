import cv2
from datetime import datetime
import logging
import math
import numpy as np
import os
import random
from shutil import get_terminal_size
import sys
import time
from thop import profile, clever_format
import torch
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR_warmup(_LRScheduler):
    def __init__(self, args, optimizer, base_lr, last_epoch=-1, min_lr=1e-7):
        self.base_lr = base_lr
        self.min_lr = min_lr
        ####The duration of the warm-up
        self.w_iter = args.WARM_UP_ITER

        self.w_fac = args.WARM_UP_FACTOR
        self.T_period = args.T_PERIOD
        self.last_restart = 0
        self.T_max = self.T_period[0]
        assert args.MAX_ITER == self.T_period[-1], 'Illegal training period setting.'
        super(CosineAnnealingLR_warmup, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        ### cosine lr
        if self.last_epoch - self.last_restart < self.w_iter:
            ratio = self.w_fac + (1 - self.w_fac) * (self.last_epoch - self.last_restart) / self.w_iter
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]

        ### warm up for a period time
        elif self.last_epoch in self.T_period:
            self.last_restart = self.last_epoch
            if self.last_epoch != self.T_period[-1]:
                self.T_max = self.T_period[self.T_period.index(self.last_epoch) + 1]
            return [self.min_lr for group in self.optimizer.param_groups]
        else:
            ratio = 1 / 2 * (1 + math.cos(
                (self.last_epoch - self.last_restart - self.w_iter) / (self.T_max - self.last_restart - self.w_iter) * math.pi))
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    # 4D: grid (B, C, H, W), 3D: (C, H, W), 2D: (H, W)
    # output_channel: bgr
    
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def calculate_cost(model, input_size=(1, 3, 224, 224)):
    input_ = torch.randn(input_size).cuda()
    macs, params = profile(model, inputs=(input_, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("MACs:" + macs + ", Params:" + params)

      
def tensor2yuv(tensor_img):
    img_y = 0.299 * tensor_img[:, 0:1, :, :] + 0.587*tensor_img[:, 1:2, :, :] + 0.114*tensor_img[:, 2:3, :, :]
    img_u = 0.492 * (tensor_img[:, 1:2, :, :] - img_y)
    img_v = 0.877 * (tensor_img[:, 0:1, :, :] - img_y)
    img_yuv = torch.cat([img_y, img_u, img_v], dim=1)

    return img_yuv


def yuv2tensor(img_yuv):
    img_r = img_yuv[:, 0:1, :, :] + 1.14*img_yuv[:, 2:3, :, :]
    img_g = img_yuv[:, 0:1, :, :] -0.39*img_yuv[:,1:2,:,:]-0.58*img_yuv[:,2:3,:,:]
    img_b = img_yuv[:,0:1,:,:]+2.03*img_yuv[:,1:2,:,:]
    img_rgb = torch.cat([img_r,img_g,img_b], dim=1)
    return img_rgb