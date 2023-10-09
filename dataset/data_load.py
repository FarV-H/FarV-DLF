import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import glob

import numpy as np
import torch
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
import pdb

class GetDataset_Mat(Dataset):
    def __init__(self, data_dir: str, size):
        self.images_dir     = []
        self.size           = size

        if not Path(data_dir).is_dir():
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        else:
            patients_dir = [x for x in Path(data_dir).iterdir()]

        for i in range(len(patients_dir)):
            self.images_dir.extend(list(Path(patients_dir[i], 'img_mat').glob('*.mat')))
        logging.info(f'Creating dataset with {len(self.images_dir)} examples')

    def __len__(self):
        return len(self.images_dir)

    @staticmethod
    def preprocess(img):
        img = img * 20

        return img

    @staticmethod
    def load(filename):
        move_ndarray        = sio.loadmat(filename)['move']
        label_ndarray       = sio.loadmat(filename)['label']

        # # 第一维度作为通道
        # if move_ndarray.ndim == 2:
        #     move_ndarray    = move_ndarray[np.newaxis, ...]
        # else:
        #     move_ndarray    = move_ndarray.transpose((2, 0, 1))
        # if label_ndarray.ndim == 2:
        #     label_ndarray   = label_ndarray[np.newaxis, ...]
        # else:
        #     label_ndarray   = label_ndarray.transpose((2, 0, 1))
        return move_ndarray, label_ndarray

    def __getitem__(self, idx):

        file_dir    = self.images_dir[idx]

        slice_idx   = int(file_dir.stem)

        move        = np.zeros([800, 800, self.size[2]])
        label       = np.zeros([800, 800, self.size[2]])
        # pdb.set_trace()
        if Path(file_dir.parent, '%03d'%(slice_idx + self.size[2] - 1) + file_dir.suffix).is_file():
            for i in range(self.size[2]):
                move_slice, label_slice     = self.load(Path(file_dir.parent, '%03d'%(slice_idx + i) + file_dir.suffix))
                move[:, :, i]               = self.preprocess(move_slice)
                label[:, :, i]              = self.preprocess(label_slice)
        else:
            for i in range(self.size[2]):
                move_slice, label_slice     = self.load(Path(file_dir.parent, '%03d'%(slice_idx + i - self.size[2] + 1) + file_dir.suffix))
                move[:, :, i]               = self.preprocess(move_slice)
                label[:, :, i]              = self.preprocess(label_slice)

        return {
            'Move': torch.as_tensor(move.copy()).unsqueeze(0).float().contiguous(),
            'Label': torch.as_tensor(label.copy()).unsqueeze(0).float().contiguous()
        }

