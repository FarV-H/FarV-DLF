import argparse
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy
import scipy.io as sio

from models import UNet
from utils.data_load import GetDataset_Mat

import pdb

def get_args():
    parser = argparse.ArgumentParser(description='Restore images from LD&SV(Low-Dose & Sparse-Views) images')
    parser.add_argument('--model', '-m', default='checkpoints\\', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--dir_input', type=str, default='D:\\data\\LD_sparse_scan\\chest\\', help='Input file directory')
    parser.add_argument('--dir_output', '-o', type=str, default='.\\test\\', help='Output file directory')
    parser.add_argument('--dir_target', type=str, default='D:\\data\\LD_sparse_scan\\chest\\', help='Target file directory')
    parser.add_argument('--ld_sv_type', type=str, default='1_4', help='Chosse low-dose CT type')
    parser.add_argument('--net_frame', type=str, default='unet', help='Chosse network frame type')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

def test_net(args,
              net,
              criterion,
              device):
    # 1. Create dataset
    dataset         = GetDataset_Mat(args.dir_input, args.dir_target)
    loader_args     = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader     = DataLoader(dataset, shuffle=False, **loader_args)

    # 5. Begin test
    net.eval()
    n_test = len(test_loader)
    loss = 0
    LD_SV = numpy.zeros([512, 512, n_test])
    ND = numpy.zeros([512, 512, n_test])
    output = numpy.zeros([512, 512, n_test])

    # iterate over the test set
    idx = 0
    for batch in tqdm(test_loader, total=n_test, desc='Test round', unit='images'):
        image, label = batch['LD'], batch['ND']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the image
            images_pred = net(image)
            loss += criterion(images_pred, label)
            LD_SV[:, :, idx]    = image.squeeze().cpu().numpy()
            ND[:, :, idx]       = label.squeeze().cpu().numpy()
            output[:, :, idx]   = images_pred.squeeze().cpu().numpy()
        idx += 1

    # save
    Path(args.dir_output).mkdir(parents=True, exist_ok=True)
    save_dir = args.dir_output + args.net_frame + '_' + args.ld_sv_type + '.mat'
    sio.savemat(save_dir, {'LD_SV':LD_SV, 'ND':ND, 'output':output})
    logging.info(f'Results saved in {save_dir}')

    # Fixes a potential division by zero error
    # Logging Loss
    if n_test == 0:
        # Logging Loss
        logging.info('Test Loss: {}'.format(loss))
        return loss
    logging.info('Test Loss: {}'.format(loss / n_test))
    return loss / n_test

if __name__ == '__main__':
    args                = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Set your data path
    input_idx           = ['A031']
    model_dir           = args.model + args.net_frame + '_' + args.ld_sv_type + '/checkpoint_epoch_100.pth'
    args.dir_input      = [Path(args.dir_input) / x / 'image' / args.ld_sv_type for x in input_idx]
    args.dir_target     = [Path(args.dir_target) / x / 'image/1_1' for x in input_idx]

    # Load your model
    if args.net_frame == 'unet':
        net             = UNet(in_channels=1, out_channels=1, bilinear=args.bilinear)

    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_dir}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model_dir, map_location=device))

    logging.info('Model loaded!')

    criterion   = nn.MSELoss()
    test_net(args=args, net=net, criterion=criterion, device=device)
