import os
import torch
import argparse
import logging
import wandb
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time

from dataset.data_load import GetDataset_Mat
from evaluate import evaluate
from models import *
from losses import *

import pdb

def get_args():
    parser = argparse.ArgumentParser(description='Train the network on images and targets')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=400, help='Number of epochs')
    parser.add_argument('--epochs_step', '-s', metavar='S', type=int, default=1000, help='Number of steps each epoch')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.0005, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--net_frame', type=str, default='U-Net_3D', help='Chosse network frame type')
    parser.add_argument('--dir_input', type=str, default='/mnt/kunlun/users/hfw/PZ_HP/data/FS_simulate_data_0924/', help='Input file directory')
    parser.add_argument('--dir_target', type=str, default='/mnt/kunlun/users/hfw/PZ_HP/data/FS_simulate_data_0924/', help='Target file directory')
    parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints/', help='Input file directory')
    
    return parser.parse_args()

def train_net(args, net, device, save_checkpoint: bool = True):
    # 1. Create dataset
    dataset = GetDataset_Mat(args.dir_input, args.size)

    # 2. Split into train / validation partitions
    n_val   = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    if args.epochs_step == 0:
        args.epochs_step = n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args         = dict(batch_size=args.batch_size, num_workers=1, pin_memory=True)
    train_loader        = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader          = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment          = wandb.init(project=args.net_frame, resume='allow')
    experiment.config.update(dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, val_percent=args.val, save_checkpoint=save_checkpoint, amp=args.amp))

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {args.amp}''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer           = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler           = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    grad_scaler         = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion           = LossFunc().cuda()
    global_step         = 0

    # 5. Begin training
    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss      = 0
        step            = 0
        with tqdm(total=args.epochs_step, desc=f'Epoch {epoch}/{args.epochs}', unit='img', ncols=120) as pbar:
            for batch in train_loader:
                move            = batch['Move']
                label           = batch['Label']

                if args.size[0] >= 800:
                    x_idx       = 0
                else:
                    x_idx       = np.random.randint(0, 800 - args.size[0] - 1)
                if args.size[1] >= 800:
                    y_idx       = 0
                else:
                    y_idx       = np.random.randint(0, 800 - args.size[1] - 1)

                move            = move[:, :, x_idx:x_idx + args.size[0], y_idx:y_idx + args.size[1], :]
                label           = label[:, :, x_idx:x_idx + args.size[0], y_idx:y_idx + args.size[1], :]

                assert move.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {move.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                move            = move.to(device=device, dtype=torch.float32)
                label           = label.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    images_pred = net(move)
                    loss        = criterion(images_pred, label)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(move.shape[0])
                step            += 1
                global_step     += 1
                epoch_loss      += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if step == args.epochs_step:
                    break

        if global_step % args.epochs_step == 0:
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag]    = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag]  = wandb.Histogram(value.grad.data.cpu())
            
            move_eval, label_eval, images_pred_eval, loss_score = evaluate(net, val_loader, criterion, args, device)
            scheduler.step(loss_score)
            
            logging.info('Validation Loss: {}'.format(loss_score))

            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Loss': loss_score,
                'Move': wandb.Image(move_eval.squeeze().cpu().numpy()[:, :, 31]),
                'Label': wandb.Image(label_eval.squeeze().cpu().numpy()[:, :, 31]),
                'pred': wandb.Image(images_pred_eval.squeeze().cpu().detach().numpy()[:, :, 31]),
                'epoch': epoch,
                **histograms
            })
                
        if save_checkpoint & (epoch % 10 == 0):
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), args.dir_checkpoint + '/checkpoint_epoch_{}.pth'.format(epoch))
            logging.info(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    print(torch.__version__)
    os.environ['WANDB_MODE'] = 'online'

    args = get_args()
    print(args)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Set your data path
    network_name        = args.net_frame + '_' + time.strftime('%Y%m%d%H%M', time.localtime())
    args.dir_checkpoint = args.dir_checkpoint + network_name
    # args.dir_input      = [Path(args.dir_input) / x / 'image' / args.ld_sv_type for x in input_idx]
    # args.dir_target     = [Path(args.dir_target) / x / 'image/1_1' for x in input_idx]

    # Select your model
    args.size           = [256, 256, 32]
    if args.net_frame == 'U-Net_3D':
        net             = UNet_3D(in_channels=1, out_channels=1, bilinear=args.bilinear)
    else:
        RuntimeError(f'No net_frame found in {args.net_frame}, make sure your network frame chosen.')

    logging.info(f'Network: {args.net_frame}\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.out_channels} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(args=args, net=net, device=device)
    except KeyboardInterrupt:
        Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.dir_checkpoint + '/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise