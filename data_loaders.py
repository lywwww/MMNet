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


def data_generator_np(training_files, subject_files, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files)
    test_dataset = LoadDataset_from_numpy(subject_files)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts




def train_loader_generator(training_files, batch_size=128):
    train_dataset = LoadDataset_from_numpy(training_files)

    all_ys = train_dataset.y_data
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_size = int(len(all_ys) * 0.7)
    test_size = len(all_ys) - train_size

    train_set, test_set = random_split(
        train_dataset,
        [train_size, test_size],
        generator=torch.manual_seed(0)
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts

def test_generator(testing_files, batch_size=128):
    test_dataset = LoadDataset_from_numpy(testing_files)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return test_loader


if __name__ == '__main__':
    # train_loader, counts = train_loader_generator(
    #     ['/home/deep1/17145213/data/edf20/edf20.npz'], 128)
    train_loader, test_loader, counts = test_generator(
        ['/home/deep1/17145213/data/edf197/ST_all.npz'], 128)

    print(counts)


