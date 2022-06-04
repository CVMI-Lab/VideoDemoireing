import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
import numpy as np
import torchvision
import cv2
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from model import MainNet, PcdAlign


class VDM_PCD(nn.Module):
    def __init__(self, args, pretrain=False, freeze=False):
        super(VDM_PCD, self).__init__()
        self.args = args
        self.use_shuffle = args.use_shuffle
        self.backbone = args.backbone
        self.num_res_blocks = list(map(int, args.num_res_blocks.split('+')))

        # PCD align
        self.pcdalign = PcdAlign.PcdAlign(nf=args.n_feats)

        # Demoireing backbone
        if self.backbone == 'vdm_pcd_v1':
            self.MainNet_V1 = MainNet.MainNet_V1(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats,
                                                 res_scale=args.res_scale, use_shuffle=args.use_shuffle)
        else:
            # Alternatively, other SOTA demoireing models can be adopted.
            print('Replace with your own demoireing model!')
            import pdb; pdb.set_trace()

        # # If true, load pretrained weights trained on other demoire dataset.
        # if pretrain:
        #     pre_trained_dir = 'pre_trained/vdm/shuffle_49.pth'
        #     if not self.use_shuffle:
        #         pre_trained_dir = 'pre_trained/vdm/noshuffle_49.pth'
        #         pre_trained_weights = torch.load(pre_trained_dir)['state_dict']
        #         self.MainNet_V1.load_state_dict(pre_trained_weights, strict=False)
        #         print('initialize: %s' % pre_trained_dir)
        #
        #     # freeze the demoireing weights
        #     if freeze:
        #         to_freeze_names = pre_trained_weights.keys()
        #         for name, param in self.MainNet_V1.named_parameters():
        #             if name in to_freeze_names:
        #                 param.requires_grad = False
        #         print('freeze pre-trained params')

        # RGB image with 3 channels
        if self.use_shuffle:
            in_channels = 12
        else:
            in_channels = 3

        # generate multi-level features for demoireing
        self.conv1 = nn.Conv2d(in_channels, args.n_feats, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(args.n_feats, 2*args.n_feats, 3, 2, 1, bias=True)
        self.conv2_1 = nn.Conv2d(2*args.n_feats, args.n_feats, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(args.n_feats, 2*args.n_feats, 3, 2, 1, bias=True)
        self.conv3_1 = nn.Conv2d(2*args.n_feats, args.n_feats, 1, 1, 0, bias=True)

        # aggregate/blend multi-level aligned features
        self.conv_blend_lv1 = nn.Conv2d(args.n_feats * (args.NUM_AUX_FRAMES + 1), args.NUM_AUX_FRAMES + 1, 3, 1, 1, bias=True)
        self.conv_channel_lv1 = nn.Conv2d(args.n_feats, args.n_feats, 1, 1, 0, bias=True)
        self.conv_blend_lv2 = nn.Conv2d(args.n_feats * (args.NUM_AUX_FRAMES + 1), args.NUM_AUX_FRAMES + 1, 3, 1, 1, bias=True)
        self.conv_channel_lv2 = nn.Conv2d(args.n_feats, args.n_feats, 1, 1, 0, bias=True)
        self.conv_blend_lv3 = nn.Conv2d(args.n_feats * (args.NUM_AUX_FRAMES + 1), args.NUM_AUX_FRAMES + 1, 3, 1, 1, bias=True)
        self.conv_channel_lv3 = nn.Conv2d(args.n_feats, args.n_feats, 1, 1, 0, bias=True)

    def down_shuffle(self, x, r):
        b, c, h, w = x.size()
        out_channel = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        x = x.view(b, c, out_h, r, out_w, r)
        out = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)
        return out

    def forward(self, cur=None, ref=None, label=None, blend=1):
        # pixel shuffle
        if self.use_shuffle:
            cur = self.down_shuffle(cur, 2)

        # generate multi-level features for PCD
        cur_lv1 = self.conv1(cur)
        cur_lv1_1 = F.interpolate(cur_lv1, scale_factor=0.5, mode='bilinear', align_corners=False)
        cur_lv1_2 = F.interpolate(cur_lv1, scale_factor=0.25, mode='bilinear', align_corners=False)
        cur_feats = [cur_lv1, cur_lv1_1, cur_lv1_2]

        # generate multi-level features for demoireing
        cur_lv2 = self.conv2_1(self.conv2(cur_lv1))
        cur_lv3 = self.conv3_1(self.conv3(cur_lv2))

        # aligned features
        align_feats_lv1 = cur_lv1
        align_feats_lv2 = cur_lv2
        align_feats_lv3 = cur_lv3

        # align reference/nearby frames to current frame
        for i in range(self.args.NUM_AUX_FRAMES):
            # extract features from reference images
            ref_tmp = ref[:, (0 + 3 * i):(3 + 3 * i), :, :]
            if self.use_shuffle:
                ref_tmp = self.down_shuffle(ref_tmp, 2)
            ref_lv1 = self.conv1(ref_tmp)
            ref_lv1_1 = F.interpolate(ref_lv1, scale_factor=0.5, mode='bilinear', align_corners=False)
            ref_lv1_2 = F.interpolate(ref_lv1, scale_factor=0.25, mode='bilinear', align_corners=False)
            ref_feats = [ref_lv1, ref_lv1_1, ref_lv1_2]

            # align features using pcd
            T_lv1 = self.pcdalign(nbr_fea_l=ref_feats, ref_fea_l=cur_feats)

            # generate multi-level features for demoireing
            T_lv2 = self.conv2_1(self.conv2(T_lv1))
            T_lv3 = self.conv3_1(self.conv3(T_lv2))

            # concatenate features
            align_feats_lv1 = torch.cat((align_feats_lv1, T_lv1), dim=1)
            align_feats_lv2 = torch.cat((align_feats_lv2, T_lv2), dim=1)
            align_feats_lv3 = torch.cat((align_feats_lv3, T_lv3), dim=1)

        # merge features
        if blend == 1:  # use predicted blending weights
            weight_lv1 = F.softmax(self.conv_blend_lv1(align_feats_lv1), dim=1)
            weight_lv2 = F.softmax(self.conv_blend_lv2(align_feats_lv2), dim=1)
            weight_lv3 = F.softmax(self.conv_blend_lv3(align_feats_lv3), dim=1)

            # # save blending weights
            # torchvision.utils.save_image(weight_lv3[:,0:1,:,:].detach().cpu(), 'weight_lv3_0.png')
            # torchvision.utils.save_image(weight_lv2[:,0:1,:,:].detach().cpu(), 'weight_lv2_0.png')
            # torchvision.utils.save_image(weight_lv1[:,0:1,:,:].detach().cpu(), 'weight_lv1_0.png')

            merge_feats_lv1 = align_feats_lv1[:, 0:self.args.n_feats, :, :] * weight_lv1[:, 0:1, :, :]
            merge_feats_lv2 = align_feats_lv2[:, 0:self.args.n_feats, :, :] * weight_lv2[:, 0:1, :, :]
            merge_feats_lv3 = align_feats_lv3[:, 0:self.args.n_feats, :, :] * weight_lv3[:, 0:1, :, :]

            for j in range(1, 1 + self.args.NUM_AUX_FRAMES):
                merge_feats_lv1 = merge_feats_lv1 + align_feats_lv1[:, (self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :, :] * weight_lv1[:, j:(j + 1), :, :]
                merge_feats_lv2 = merge_feats_lv2 + align_feats_lv2[:, (self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :, :] * weight_lv2[:, j:(j + 1), :, :]
                merge_feats_lv3 = merge_feats_lv3 + align_feats_lv3[:, (self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :, :] * weight_lv3[:, j:(j + 1), :, :]

            #     # save blending weights, auxiliary frames
            #     torchvision.utils.save_image(weight_lv3[:,j:(j+1),:,:].detach().cpu(), 'weight_lv3_%s.png' % j)
            #     torchvision.utils.save_image(weight_lv2[:,j:(j+1),:,:].detach().cpu(), 'weight_lv2_%s.png' % j)
            #     torchvision.utils.save_image(weight_lv1[:,j:(j+1),:,:].detach().cpu(), 'weight_lv1_%s.png' % j)
            # import pdb; pdb.set_trace()

        elif blend == 2:
            # average the aligned features
            merge_feats_lv1 = align_feats_lv1[:, 0:self.args.n_feats, :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
            merge_feats_lv2 = align_feats_lv2[:, 0:self.args.n_feats, :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
            merge_feats_lv3 = align_feats_lv3[:, 0:self.args.n_feats, :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)

            for j in range(1, 1 + self.args.NUM_AUX_FRAMES):
                merge_feats_lv1 = merge_feats_lv1 + align_feats_lv1[:, (self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
                merge_feats_lv2 = merge_feats_lv2 + align_feats_lv2[:, (self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)
                merge_feats_lv3 = merge_feats_lv3 + align_feats_lv3[:, (self.args.n_feats * j):(self.args.n_feats + self.args.n_feats * j), :, :] * 1 / (1 + self.args.NUM_AUX_FRAMES)

        else:
            print('Provide your own blending method')
            import pdb; pdb.set_trace()

        # refine the merged features
        merge_feats_lv1 = self.conv_channel_lv1(merge_feats_lv1)
        merge_feats_lv2 = self.conv_channel_lv2(merge_feats_lv2)
        merge_feats_lv3 = self.conv_channel_lv3(merge_feats_lv3)

        # demoireing
        dm_lv3, dm_lv2, dm_lv1, f_lv1, f_lv2, f_lv3 = self.MainNet_V1(merge_feats_lv3, merge_feats_lv2, merge_feats_lv1)

        return dm_lv3, dm_lv2, dm_lv1, f_lv1, f_lv2, f_lv3


def warp(x, flo):
    """
    From PWCNet, warp an image/tensor (im2) back to im1, according to the optical flow
    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W], pre-computed optical flow, im2-->im1
    Returns: warped image and mask indicating valid positions
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask, mask


###############################################################################################
def model_fn_decorator(loss_fn, mode='train'):

    # vgg features as region-level statistics
    if mode == 'train':
        vgg_loss = VGGPerceptualLoss().cuda()

    def val_model_fn(args, data, model, net_metric, save_path):
        model.eval()

        # prepare input and forward
        number = data['number']
        in_img = data['in_img'][0].cuda()
        label = data['label'][0].cuda()
        num_img_aux = len(data['in_img_aux'])
        assert num_img_aux > 0
        in_img_aux = data['in_img_aux'][0].cuda()
        for i in range(1, num_img_aux):
            in_img_aux = torch.cat([in_img_aux, data['in_img_aux'][i].cuda()], axis=1)

        with torch.no_grad():
            out3, out2, out_img, _, _, _ = model(cur=in_img, ref=in_img_aux)
            loss = loss_fn(out3, out2, out_img, label, feature_layers=[2])

        out_put = tensor2img(out_img)
        gt = tensor2img(label)
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(label, min=0, max=1)

        # Calculate LPIPS
        cur_lpips = net_metric.forward(pre, tar, normalize=True)
        # Calculate PSNR
        cur_psnr = calculate_psnr(out_put, gt)
        # Calculate SSIM
        cur_ssim = calculate_ssim(out_put, gt)

        # save images
        if args.SAVE_IMG != 0:
            out_save = out_img.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'val_%05s' % number[0] + '.%s' % args.SAVE_IMG)

        return loss, cur_psnr, cur_ssim, cur_lpips.item()


    def test_model_fn(args, data, model, save_path):
        model.eval()

        # prepare input and forward
        number = data['number']
        in_img = data['in_img'][0].cuda()
        num_img_aux = len(data['in_img_aux'])
        assert num_img_aux > 0
        in_img_aux = data['in_img_aux'][0].cuda()
        for i in range(1, num_img_aux):
            in_img_aux = torch.cat([in_img_aux, data['in_img_aux'][i].cuda()], axis=1)

        with torch.no_grad():
            out3, out2, out_img, _, _, _ = model(cur=in_img, ref=in_img_aux)

        # save images
        if args.SAVE_IMG != 0:
            out_save = out_img.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'test_%05s' % number[0] + '.%s' % args.SAVE_IMG)


    def train_model_fn(args, data, model, iters, epoch):
        model.train()

        # prepare input and forward
        if args.use_temporal and (epoch >= args.temporal_begin_epoch):
            # currently, only support 2 branches
            in_img = data['in_img'][0].cuda()
            label = data['label'][0].cuda()
            img_aux_list = data['in_img_aux'][0:args.NUM_AUX_FRAMES]
            in_img_1 = data['in_img'][1].cuda()
            label_1 = data['label'][1].cuda()
            img_aux_list1 = data['in_img_aux'][args.NUM_AUX_FRAMES:2 * args.NUM_AUX_FRAMES]

            in_img_aux = img_aux_list[0].cuda()
            in_img_aux_1 = img_aux_list1[0].cuda()
            for i in range(1, args.NUM_AUX_FRAMES):
                in_img_aux = torch.cat([in_img_aux, img_aux_list[i].cuda()], axis=1)
                in_img_aux_1 = torch.cat([in_img_aux_1, img_aux_list1[i].cuda()], axis=1)

            out3, out2, out_img, f_lv1, f_lv2, f_lv3 = model(cur=in_img, ref=in_img_aux)
            out3_1, out2_1, out_img_1, f_lv1_1, f_lv2_1, f_lv3_1 = model(cur=in_img_1, ref=in_img_aux_1)

            # reconstruction loss
            loss = loss_fn(out3, out2, out_img, label, feature_layers=[2]) + loss_fn(out3_1, out2_1, out_img_1, label_1, feature_layers=[2])
            loss_reg = 0 * loss
            loss_temporal = 0 * loss

            # temporal consistency loss
            ## basic relation-based loss
            if args.temporal_loss_mode == 0:
                # regress output_error to gt_error upon pixel-level
                gt_error = label - label_1
                out_error = out_img - out_img_1
                loss_temporal = F.l1_loss(gt_error, out_error)

            ## use multi-scale relation-based loss
            elif args.temporal_loss_mode == 1:
                # blur image/area statistics/intensity
                # k_sizes = [1, 3, 5, 7]
                k_sizes = args.k_sizes
                gt_errors = []
                out_errors = []

                for i in range(len(k_sizes)):
                    k_size = k_sizes[i]
                    avg_blur = nn.AvgPool2d(k_size, stride=1, padding=int((k_size - 1) / 2))
                    gt_error = avg_blur(label) - avg_blur(label_1)
                    out_error = avg_blur(out_img) - avg_blur(out_img_1)
                    gt_errors.append(gt_error)
                    out_errors.append(out_error)

                gt_error_rgb_pixel_min = gt_errors[0]
                out_error_rgb_pixel_min = out_errors[0]

                for j in range(1, len(k_sizes)):
                    gt_error_rgb_pixel_min = torch.where(torch.abs(out_error_rgb_pixel_min) < torch.abs(out_errors[j]),
                            gt_error_rgb_pixel_min, gt_errors[j])
                    out_error_rgb_pixel_min = torch.where(torch.abs(out_error_rgb_pixel_min) < torch.abs(out_errors[j]),
                            out_error_rgb_pixel_min, out_errors[j])

                loss_temporal = F.l1_loss(gt_error_rgb_pixel_min, out_error_rgb_pixel_min)

            ## Alternatively, combine relation-based loss at different scales with different weights
            elif args.temporal_loss_mode == 2:
                # blur image/area statistics/intensity
                # k_sizes = [1, 3, 5, 7]
                k_sizes = args.k_sizes
                # k_weights = [0.25, 0.25, 0.25, 0.25]
                k_weights = args.k_weights
                loss_temporal = 0*loss

                for i in range(len(k_sizes)):
                    k_size = k_sizes[i]
                    k_weight = k_weights[i]
                    avg_blur = nn.AvgPool2d(k_size, stride=1, padding=int((k_size - 1) / 2))
                    gt_error = avg_blur(label) - avg_blur(label_1)
                    out_error = avg_blur(out_img) - avg_blur(out_img_1)
                    loss_temporal = loss_temporal + F.l1_loss(gt_error, out_error) * k_weight

            # ## use traditional flow-based loss
            # elif args.temporal_loss_mode == 3:
            #     # make sure you have correctly pre-computed the flow and occ_mask, flow: img1-->img
            #     flow = data['flow'][0].cuda()
            #     occu_mask = data['mask'][0].cuda()[:, 0:1, :, :]
            #     # image warp
            #     out_img_1_warped, mask_boundary = warp(out_img_1, flow)
            #     label_1_warped, mask_boundary = warp(label_1, flow)
            #     occu_mask = occu_mask * mask_boundary
            #
            #     # # save warped image
            #     # torchvision.utils.save_image(out_img[:,0:1,:,:].detach().cpu(), 'img.png')
            #     # torchvision.utils.save_image(out_img_1_warped[:,0:1,:,:].detach().cpu(), 'img_1_warped.png')
            #
            #     loss_temporal = F.l1_loss(out_img * occu_mask, out_img_1_warped * occu_mask)

            # ## calculate relation-based loss in VGG feature domain
            # elif args.temporal_loss_mode == 4:
            #     label_feature, label_1_feature = vgg_loss(label, label_1, feature_layers=[2],
            #                                               mask=None, return_feature=True)
            #     out_img_feature, out_img_1_feature = vgg_loss(out_img, out_img_1, feature_layers=[2],
            #                                                   mask=None, return_feature=True)
            #
            #     out_feature_error = out_img_feature - out_img_1_feature
            #     gt_feature_error = label_feature - label_1_feature
            #
            #     loss_temporal = F.l1_loss(gt_feature_error, out_feature_error)

            else:
                loss_temporal = 0*loss

            loss_temporal = args.weight_t * loss_temporal
            loss = loss + loss_temporal

        # do not use temporal constraints
        else:
            in_img = data['in_img'][0].cuda()
            label = data['label'][0].cuda()
            num_img_aux = len(data['in_img_aux'])
            in_img_aux = data['in_img_aux'][0].cuda()
            for i in range(1, args.NUM_AUX_FRAMES):
                in_img_aux = torch.cat([in_img_aux, data['in_img_aux'][i].cuda()], axis=1)

            out3, out2, out_img, f_lv1, f_lv2, f_lv3 = model(cur=in_img, ref=in_img_aux)
            loss = loss_fn(out3, out2, out_img, label, feature_layers=[2])

            # # do not use deep supervision loss
            # loss = loss_fn(out_img, label, feature_layers=[2])

            loss_temporal = loss*0
            loss_reg = loss*0

        # save images
        if iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
            in_save = in_img.detach().cpu()[:, 0:3, :, :]
            out_save = out_img.detach().cpu()
            gt_save = label.detach().cpu()
            res_save = torch.cat((in_save, out_save, gt_save), 2)
            save_number = (iters + 1) // args.SAVE_ITER
            torchvision.utils.save_image(res_save, args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg')

        return loss, loss_temporal, loss_reg

    if mode == 'test':
        fn = test_model_fn
    elif mode == 'val':
        fn = val_model_fn
    else:
        fn = train_model_fn
    return fn


















