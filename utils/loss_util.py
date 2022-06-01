import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *


class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam_p=1.0, lam_l=0.5):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam_p = lam_p
        self.lam_l = lam_l

    def forward(self, out3, out2, out1, gt1, feature_layers=[2], mask=None):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        if mask is not None:
            mask2 = F.interpolate(mask, scale_factor=0.5, mode='bilinear', align_corners=False)
            mask3 = F.interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=False)
            loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers, mask=mask) + self.lam_l*F.l1_loss(out1*mask, gt1*mask)
            loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers, mask=mask2) + self.lam_l*F.l1_loss(out2*mask2, gt2*mask2)
            loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers, mask=mask3) + self.lam_l*F.l1_loss(out3*mask3, gt3*mask3)
        else: 
            loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out1, gt1)
            loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out2, gt2)
            loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out3, gt3)
        return loss1+loss2+loss3
        

class single_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam_p=1.0, lam_l=0.5):
        super(single_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam_p = lam_p
        self.lam_l = lam_l

    def forward(self, out, gt, feature_layers=[2], mask=None):
        if mask is not None:
            loss = self.lam_p*self.loss_fn(out, gt, feature_layers=feature_layers, mask=mask) + self.lam_l*F.l1_loss(out*mask, gt*mask)
        else:
            loss = self.lam_p*self.loss_fn(out, gt, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out, gt)
        return loss
        

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if mask is not None:
                    _,_,H,W = x.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')[:, 0:1, :, :]
                    x = x*mask_resized
                    y = y*mask_resized
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += torch.nn.functional.l1_loss(x, y)
                    
                if return_feature:
                    return x, y
                    
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
        
        
        
        
