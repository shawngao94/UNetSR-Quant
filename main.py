from __future__ import print_function

import argparse

from torch.utils.data import DataLoader
from Unet.solver import unetTrainer
from dataset import DatasetFromFolder


# ===========================================================
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize','-b', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--crop_size', '-cs',  type=int, default=128, help="crop size during training")
parser.add_argument('--nEpochs','-n', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', '-lr',type=float, default=0.001, help='Learning Rate. Default=0.001')
# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='unetQ', help='choose which model is going to use')
parser.add_argument('--nbits', type=int, default=8, help='bits of quantization')
# path
parser.add_argument('--train_data_path', type=str, default=r'datasets\BSDS300\images\train', help='data path for training')
parser.add_argument('--test_data_path', type=str, default=r'datasets\Set5', help='data path for test')
parser.add_argument('--fp32_model_path', type=str, default=None, help='fp32 pre-trained model path')
parser.add_argument('--load_path', type=str, default=None, help='model training checkpoints')

args = parser.parse_args()


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = DatasetFromFolder(args.train_data_path, args.upscale_factor, args.crop_size)
    test_set = DatasetFromFolder(args.test_data_path, args.upscale_factor, None)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    
    if args.model == 'unet':
        model = unetTrainer(args, training_data_loader, testing_data_loader, quan=False)
    elif args.model == 'unetQ':
        model = unetTrainer(args, training_data_loader, testing_data_loader, quan=True)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    main()
  