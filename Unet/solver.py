from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

#from FilterCNN.model import Net
from progress_bar import progress_bar

from Unet.Umodel import UNet2
from Unet.GraLoss import GradientLoss

import Unet.pytorch_ssim as pytorch_ssim
import numpy as np
from PIL import Image
import os
from Unet.unet_parts import *
from torch.utils.tensorboard import SummaryWriter

class unetTrainer(object):
    def __init__(self, config, training_loader, testing_loader, quan=False):
        super(unetTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.mse = None
        self.optimizer = None
        self.scheduler = None
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.quan = quan
        self.root = 'unetQ' if self.quan else 'unet'
        self.fp32_model_path = config.fp32_model_path
        self.nbits_weight = config.nbits
        self.nbits_act = config.nbits
        self.load_path = config.load_path

    def build_model(self):
        if self.upscale_factor==2:
            self.model=UNet2(3,3, self.quan, self.nbits_weight, self.nbits_act).to(self.device)
        self.model.weight_init()
        if self.load_path:
            try:
                self.model.load_state_dict(torch.load(self.load_path))
            except:
                self.model.load_state_dict(torch.load(self.load_path).state_dict())
            print("Model has been loaded from %s" % self.load_path)
        elif self.fp32_model_path and self.quan:
            fp32_model = torch.load(self.fp32_model_path)
            self.model.qat_from_float(fp32_model)
            del fp32_model
            print("Quantized(%s) Model has been loaded from float32 Checkpoint!" % self.quan)
        self.mse = torch.nn.MSELoss()
        self.l1=torch.nn.L1Loss()
        self.gradient_loss = GradientLoss()
        print('Model Built!\n', self.model)
        print('# model parameters:', sum(param.numel() for param in self.model.parameters()))

        if self.CUDA:
            cudnn.benchmark = True
            self.mse.cuda()
            self.l1.cuda()
            self.gradient_loss.cuda()
        if not self.quan:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.qat_set_lr(self.lr), momentum=0.9)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,150,9000], gamma=0.1)


    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            prediction = self.model(data)
            loss1 = self.l1(prediction, target)
            # loss2 = self.gradient_loss(prediction, target)
            loss = loss1#+loss2
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        return train_loss / len(self.training_loader)

    def test(self):
        self.model.eval()
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data).clamp(0, 1)
                mse = self.mse(prediction, target)
                psnr = 10 * log10(1. / mse.item())
                avg_psnr += psnr
                ssim_value = pytorch_ssim.ssim(prediction, target)
                avg_ssim += ssim_value
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f | SSIM: %.4f' % ((avg_psnr / (batch_num + 1)),avg_ssim / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
        return avg_psnr / len(self.testing_loader)

    def save(self, save_model=True):
        self.model.eval()
        if save_model:
            if not os.path.exists("output-%s/ckpt/" % self.root):
                os.makedirs("output-%s/ckpt/" % self.root)
            model_out_path = "output-%s/ckpt/model.pth" % self.root
            torch.save(self.model, model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))
        avg_psnr = 0
        avg_ssim = 0
        f = open("output-%s/result.txt" % self.root, 'w')
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data).clamp(0, 1)
                mse = self.mse(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                ssim_value = pytorch_ssim.ssim(prediction, target)
                avg_ssim += ssim_value
                f.write(str(batch_num)+": %.2f %.2f\n" % (psnr, ssim_value))
                prediction = torch.squeeze(prediction).permute(1, 2, 0).cpu().detach().numpy()
                prediction = np.uint8(prediction * 255)
                prediction = Image.fromarray(prediction, mode='RGB')
                if not os.path.exists("output-%s/images/" % self.root):
                    os.makedirs("output-%s/images/" % self.root)
                prediction.save("output-%s/images/" % self.root +str(batch_num)+".png")

                target = torch.squeeze(target).permute(1, 2, 0).cpu().detach().numpy()
                target = np.uint8(target * 255)
                target = Image.fromarray(target, mode='RGB')
                target.save("output-%s/images/" % self.root +str(batch_num)+"_gt.png")
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f | SSIM: %.4f' % ((avg_psnr / (batch_num + 1)),avg_ssim / (batch_num + 1)))
        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
        f.write("Average PSNR: {:.4f} dB\n".format(avg_psnr / len(self.testing_loader)))
        f.write("Average SSIM: {:.4f}\n".format(avg_ssim / len(self.testing_loader)))
        f.close()

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            avg_loss = self.train()
            # if epoch == 1:
                # tb_writer = SummaryWriter(log_dir='output-%s/logs' % self.root)
            if epoch == 1 or epoch % 10 == 0:
                avg_psnr = self.test()
                # tb_writer.add_scalar('psnr_eval', avg_psnr, epoch)
            # tb_writer.add_scalar('loss', avg_loss, epoch)
            self.scheduler.step()
        self.save()
