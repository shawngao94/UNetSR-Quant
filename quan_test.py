import numpy as np
import torch
from Unet.Umodel import UNet2
from PIL import Image
from torchvision import transforms
import os
import math
import Unet.pytorch_ssim as pytorch_ssim


def fp32_2_quan_model(fp32_path, qint_path):
    model = torch.load(fp32_path)
    model.eval()
    model.cpu()
    model.quant_simulation()
    torch.save(model, qint_path)
    print("fp 32 model has been converted to qint model to %s" % qint_path)
    return model

def evaluation(model, data_path):
    data_list = os.listdir(data_path)
    avg_psnr = 0
    avg_ssim = 0
    for data in data_list:
        img = Image.open(os.path.join(data_path, data))
        img = img.convert('RGB')
        img = transforms.CenterCrop(min(img.size))(img)
        img = transforms.Resize(256)(img)
        input = transforms.Resize(128)(img)
        input = transforms.ToTensor()(input).unsqueeze(0)
        img = transforms.ToTensor()(img).unsqueeze(0)
        output = model.eval_quant(input)
        mse = torch.nn.MSELoss()(output, img)
        psnr = 10 * math.log10(1 / mse.item())
        ssim = pytorch_ssim.ssim(output, img)
        avg_psnr += psnr
        avg_ssim += ssim
        print(str(data)+": %.3f %.3f" % (psnr, ssim))
        # output = transforms.ToPILImage()(output.squeeze(0).cpu().detach())
        # output.save(os.path.join('output-unetQ', data))
    avg_psnr /= len(data_list)
    avg_ssim /= len(data_list)
    print("Average PSNR: %.3f" % avg_psnr)
    print("Average SSIM: %.3f" % avg_ssim)


if __name__ == '__main__':
    fp32 = r'output-unetQ-8bit\ckpt-1\model.pth'
    qint = r'output-unetQ-8bit\ckpt-1\model_qint8.pth'
    fp32_2_quan_model(fp32,  qint)
    model = torch.load(qint)
    print('------Converting Finished------')
    print('Quant model size is %.2f MB, while FP32 model size is %.2f MB' 
          % (os.path.getsize(qint) / 1024 / 1024, os.path.getsize(fp32) / 1024 / 1024))
    print('------Evaluation Starting------')
    evaluation(model, r'S:\codes\data\SR\Test\Set5')
    print('------Evaluation Finished------')
    
