''' network architecture for EDVR '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.DCNv2.dcn_v2 import DCN_sep as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class PcdAlign(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

    def __init__(self, nf=64, groups=8, wn=None):
        super(PcdAlign, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L3_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        # self.L3_shift = ShiftAlign(nf)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
                              # extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L2_offset_conv2 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for offset
        self.L2_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        # self.L2_shift = ShiftAlign(nf)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
                              # extra_offset_mask=True)
        self.L2_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.L1_offset_conv2 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for offset
        self.L1_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        # self.L1_shift = ShiftAlign(nf)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
                              # extra_offset_mask=True)
        self.L1_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
        self.cas_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        # L3_nbr_fea = self.L3_shift(L3_offset, nbr_fea_l[2])
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        # L2_nbr_fea = self.L2_shift(L2_offset, nbr_fea_l[1])
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        # L1_nbr_fea = self.L1_shift(L1_offset, nbr_fea_l[0])
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea


