from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image
import numpy as np

from torchvision.transforms import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".JPG", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def augmentation(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomRotation(15),
        RandomHorizontalFlip(),
        RandomVerticalFlip()
            ])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor, crop_size):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        if self.crop_size:
            input_image = augmentation(self.crop_size)(input_image)
        else:
            w, h = input_image.size
            w = int(min(w, h) // 2 * 2)
            maxw = 256 # because of bad gpu
            input_image = CenterCrop(w)(input_image)
            input_image = Resize(maxw)(input_image)
        target = input_image.copy()
        w, _ = target.size
        input_image = Resize(w // self.upscale_factor)(input_image)
        input_image = ToTensor()(input_image)
        target = ToTensor()(target)
        return input_image, target

    def __len__(self):
        return len(self.image_filenames)