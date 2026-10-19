"""Data loading for the four benchmarks used in the paper (Table I).

Each dataset is read from ``<path>/<prefix>_missing_<mu>.mat`` (produced by make_incomplete.py):

    MNIST-USPS   MNIST_USPS_missing_<mu>.mat   keys X1, X2, Y, missing_matrix
    BDGP         BDGP_missing_<mu>.mat         keys X1, X2, Y, missing_matrix
    Fashion      Fashion_missing_<mu>.mat      keys X1, X2, X3, Y, missing_matrix
    HandWritten  handwritten_missing_<mu>.mat  keys X (1 x 6 cell), Y, missing_matrix

``missing_matrix`` is an N x V binary matrix in which 1 marks a missing view (i.e. 1 - M in the
notation of the paper).
"""
import os, random, sys
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from presets import canonical_dataset_name, data_file_name


class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/", missing_ratio=0.5):
        self.data_views = list()
        self.device = device  # Store device

        db = canonical_dataset_name(db)  # e.g. 'hand' -> 'HandWritten'
        file_path = os.path.join(path, data_file_name(db, missing_ratio))

        if db == "MNIST-USPS":
            mat = sio.loadmat(file_path)
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "BDGP":
            mat = sio.loadmat(file_path)
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "Fashion":
            mat = sio.loadmat(file_path)
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "HandWritten":
            mat = sio.loadmat(file_path)
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device).float()

        self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        missing_info = self.missing_matrix[index, :]

        return sub_data_views, self.labels[index], missing_info


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_eval_loader(mv_data, batch_size):
    """Sequential loader over ALL samples (no shuffling, no dropped last batch),
    so that the clustering metrics are computed on the whole dataset."""
    return torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
