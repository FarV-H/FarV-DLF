import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import matplotlib as plt

def evaluate(net, dataloader, criterion, args, device):
    net.eval()
    num_val_batches = len(dataloader)
    loss = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False, ncols=120):
        move, label = batch['Move'], batch['Label']
        
        with torch.no_grad():
            # move images and labels to correct device and type

            x_idx = 150
            y_idx = 100
            move = move[:, :, x_idx:x_idx + args.size[0], y_idx:y_idx + args.size[1], :]
            label = label[:, :, x_idx:x_idx + args.size[0], y_idx:y_idx + args.size[1], :]

            move = move.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # predict the output
            images_pred = net(move)
            loss += criterion(images_pred, label)
    # pdb.set_trace()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return move, label, images_pred, loss
    return move, label, images_pred, loss / num_val_batches
