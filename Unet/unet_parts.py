import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized
import torch.nn.quantized.functional
import torch.quantization
import torch.nn.quantized as nnq
import math


class NAObserver(torch.quantization.ObserverBase):
    # only for from_float()
    def __init__(self, dtype=torch.quint4x2, is_dynamic = False, qscheme=torch.per_tensor_affine):
        super().__init__(dtype)
        self.dtype = dtype
        self.is_dynamic = is_dynamic
        self.qscheme = qscheme

    def calculate_qparams(self):
        return 1, 0
    
    def forward(self, x):
        return x


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass_improved(x):
    input = x
    x_round = input.round()
    x = input - input.floor().detach()
    x = (torch.tanh(1*(x-0.5)) / torch.tanh(torch.ones(1).cuda()*0.5)) / 2 + 0.5
    # x = torch.tanh(2*x-1) / torch.tanh(torch.ones(1).cuda()) + 0.5
    out3 = x + input.floor().detach()
    return x_round.detach() - out3.detach() + out3

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class actQ(nn.Module):
    def __init__(self, nbits_act=4):
        super(actQ, self).__init__()
        self.nbits = nbits_act
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        # self.signed.data.fill_(sign)
        if self.signed == 1:
            self.Qn = -2 ** (self.nbits - 1)
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.nbits - 1
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def update_params(self, sign, scale=None):
        self.signed.data.fill_(sign)
        if self.signed == 1:
            self.Qn = -2 ** (self.nbits - 1)
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.nbits - 1
        if scale is not None:
            self.scale.data.copy_(scale)
            self.init_state.fill_(1)
        else:
            self.init_state.fill_(0)

    def forward(self, x):
        if self.init_state == 0:
            # Initialization method of LSQ
            # self.scale.data.copy_(2 * x.abs().mean() / math.sqrt(self.Qp))
            # Initialization method of RLQ
            # self.scale.data.copy_(x.abs().max() / (self.Qp - 1))
            minx = 0 if self.signed == 0 else x.min()
            self.scale.data.copy_((x.max() - minx) / (self.Qp - self.Qn))
            self.init_state.fill_(1)

        if self.training:
            g = 1.0 / math.sqrt(x.numel() * self.Qp)
            scale = grad_scale(self.scale, g)
            x = x / scale
            x = torch.clamp(x, self.Qn, self.Qp)
            x = round_pass(x) * scale
        else:
            x = (x / self.scale).clamp(self.Qn, self.Qp).round() * self.scale
        
        return x
    
    def eval_quant(self, x):
        self.eval()
        x = (x / self.scale).clamp(self.Qn, self.Qp).round()
        return x

class Conv2dQReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, nbits_weight=4, nbits_act=4):
        super(Conv2dQReLU, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups)
        self.act_fn = actQ(nbits_act=nbits_act)
        self.nbits_weight = nbits_weight
        
        self.Qn = -2 ** (self.nbits_weight - 1)
        self.Qp = 2 ** (self.nbits_weight - 1) - 1
        self.g = 1.0 / math.sqrt(self.conv.weight.numel() * self.Qp)
        
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.tau = nn.Parameter(torch.zeros(1, in_ch, 1, 1), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        if self.init_state == 0:
            self.tau.data.copy_(torch.zeros_like(self.tau.data))
            # for 8bit, torch.ones(1) / (self.Qp - 1)
            # for 4bit, torch.ones(1) * 0.5 / (self.Qp - 1)
            self.scale.data.copy_(2 * self.conv.weight.abs().mean() / math.sqrt(self.Qp - self.Qn))
            # Initialization method of LSQ
            # self.scale.data.copy_(2 * self.conv.weight.abs().mean() / math.sqrt(self.Qp)
            # Initialization method of RLQ
            # self.scale.data.copy_(self.conv.weight.abs().max() / (self.Qp - 1))
            self.init_state.fill_(1)
        if self.training:
            scale = grad_scale(self.scale, self.g)
            weight = self.conv.weight.data / scale
            weight = torch.clamp(weight, self.Qn, self.Qp)
            weight = round_pass(weight) * scale
        else:
            weight = (self.conv.weight.data / self.scale).clamp(self.Qn, self.Qp).round() * self.scale
        x = self.act_fn(x + self.tau)
        # x = F.relu(x)
        return F.conv2d(x, weight, self.conv.bias, self.stride, self.padding, self.dilation, self.groups)
    
    
    def eval_quant(self, x):
        with torch.no_grad():
            x = self.act_fn.eval_quant(x + self.tau)
            weight = torch.dequantize(self.conv.weight()) # pytorch needs to dequantize the weight for calculation
            bias = self.conv.bias().data
            x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups) # conv using low-bit value
            x = x * self.act_fn.scale * self.scale
        return x
        
    def quant_simulation(self):
        self.conv.add_module('weight_fake_quant', NAObserver(dtype=torch.qint8))
        self.conv.add_module('activation_post_process', NAObserver(dtype=torch.qint8))
        self.conv.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.conv.weight.data = ((self.conv.weight.data / self.scale).clamp(self.Qn, self.Qp)).round()
        self.conv.bias.data = (self.conv.bias.data / (self.scale * self.act_fn.scale)).round() # bias is saved in int32 format
        qlayer = nnq.Conv2d
        self.conv = qlayer.from_float(self.conv)


class Conv2dReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv(x))
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, quan=False, nbits_weight=4, nbits_act=4):
        super(inconv, self).__init__()
        if quan:
            self.conv = Conv2dQReLU(in_ch, out_ch, nbits_weight=nbits_weight, nbits_act=nbits_act)
            self.conv.act_fn.update_params(1)
            # self.conv.act_fn = nn.Sequential()
            self.act = nn.Sequential()
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv(x))
        return x
    
    def eval_quant(self, x):
        x = self.conv.eval_quant(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, quan=False, nbits_weight=4, nbits_act=4):
        super(down, self).__init__()
        self.mp = nn.MaxPool2d(2)
        if quan:
            self.conv = nn.Sequential(
                Conv2dQReLU(in_ch, out_ch, nbits_weight=nbits_weight, nbits_act=nbits_act),
            )
        else:
            self.conv = nn.Sequential(
                Conv2dReLU(in_ch, out_ch),
            )

    def forward(self, x):
        x = self.mp(x)
        x = self.conv(x)
        return x
    
    def eval_quant(self, x):
        x = self.mp(x)
        for m in self.conv:
            x = m.eval_quant(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, quan=False, nbits_weight=4, nbits_act=4):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
            # raise NotImplementedError
        if quan:
            self.conv = nn.Sequential(
                Conv2dQReLU(in_ch, out_ch, nbits_weight=nbits_weight, nbits_act=nbits_act),
            )
        else:
            self.conv = nn.Sequential(
                Conv2dReLU(in_ch, out_ch),
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
    def eval_quant(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        for m in self.conv:
            x = m.eval_quant(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, quan=False, nbits_weight=4, nbits_act=4):
        super(outconv, self).__init__()
        if quan:
            self.conv = Conv2dQReLU(in_ch, out_ch, kernel_size=1, padding=0, nbits_weight=nbits_weight, nbits_act=nbits_act)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def eval_quant(self, x):
        x = self.conv.eval_quant(x)
        return x

        

