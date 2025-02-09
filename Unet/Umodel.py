import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from Unet.unet_parts import *



class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, quan=False, nbits_weight=4, nbits_act=4):
        super(UNet2, self).__init__()
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)
        self.bup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.inc = inconv(n_channels, 32, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.down1 = down(32, 64, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)        
        self.down2 = down(64, 128, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.down3 = down(128, 256, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.down4 = down(256, 256, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.up1 = up(512, 128, bilinear=True, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.up2 = up(256, 64, bilinear=True, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.up3 = up(128, 32, bilinear=True, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.up4 = up(64, 16, bilinear=True, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.outc = outconv(16, n_classes, quan=quan, nbits_weight=nbits_weight, nbits_act=nbits_act)
        self.quan = quan

    def forward(self, x):
        x = self.bup(x)
        x = self.sub_mean(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.add_mean(x)
        return x
    
    def eval_quant(self, x):
        x = self.bup(x)
        x = self.sub_mean(x)
        x1 = self.inc.eval_quant(x)
        x2 = self.down1.eval_quant(x1)
        x3 = self.down2.eval_quant(x2)
        x4 = self.down3.eval_quant(x3)
        x5 = self.down4.eval_quant(x4)
        x = self.up1.eval_quant(x5, x4)
        x = self.up2.eval_quant(x, x3)
        x = self.up3.eval_quant(x, x2)
        x = self.up4.eval_quant(x, x1)
        x = self.outc.eval_quant(x)
        x = self.add_mean(x)
        return x

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, MeanShift):
                continue
            xavier_init(m)

    def quant_simulation(self):
        assert self.quan
        for n, m in self.named_modules():
            if isinstance(m, Conv2dQReLU):
                m.quant_simulation()

    def qat_from_float(self, fp32_model):
        assert self.quan
        for n, m in fp32_model.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    if n.__contains__('outc'):
                        self.outc.conv.conv.weight.data.copy_(m.weight.data)
                        self.outc.conv.conv.bias.data.copy_(m.bias.data)
                        continue
                    if n.__contains__('inc'):
                        self.inc.conv.conv.weight.data.copy_(m.weight.data)
                        self.inc.conv.conv.bias.data.copy_(m.bias.data)
                        continue
                    mq = self
                    for att in n.split('.'):
                        mq = getattr(mq, att)
                    mq.weight.data.copy_(m.weight.data)
                    mq.bias.data.copy_(m.bias.data)
    
    def qat_set_lr(self, lr):
        assert self.quan
        base_params = []
        quan_params = []
        for n, m in self.named_parameters():
            if m.requires_grad:
                if n.__contains__('weight') or n.__contains__('bias'):
                    base_params.append(m)
                else:
                    quan_params.append(m)
        return [{'params': base_params, 'lr': 0.01 * lr}, {'params': quan_params, 'lr': lr}]


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def xavier_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range=1,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1, stride=1, padding=0)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
         
