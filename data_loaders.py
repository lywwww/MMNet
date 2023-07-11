import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from torch.utils.data import random_split


class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, npz_files):
        super(LoadDataset_from_numpy, self).__init__()

        # # load files
        print(npz_files[0])
        X_train = np.load(npz_files[0])["X"]
        y_train = np.load(npz_files[0])["Y"]

        for idx, np_file in enumerate(npz_files):
            if idx == 0:
                continue
            print(npz_files[idx])
            X_train = np.vstack((X_train, np.load(np_file)["X"]))
            y_train = np.append(y_train, np.load(np_file)["Y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

        print('finish')

    def __getitem__(self, index):
        x = self.x_data[index]
        x_amplitude = torch.abs(torch.fft.fft(torch.tensor(self.x_data[index]).float()))
        return (x, x_amplitude), self.y_data[index]

    def __len__(self):
        return self.len



    print(counts)


