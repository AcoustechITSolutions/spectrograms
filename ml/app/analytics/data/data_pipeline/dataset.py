#!/usr/bin/env
from typing import Callable
import torch
from torch.utils.data import Dataset
import numpy as np
from data.data_pipeline.utils import process_rec, process_torch


class BaseDataset(Dataset):
    """
    This dataset subclass is used for reading data records
    and passing it to corresponding models in specified shape
    """
    def __init__(self, path, preproc_fn: Callable = None):
        """
        Constructor
        :param path: path to dataset file (.npy)
        :param preproc_fn: preprocessing function. Should return new data and new shape. (optional)
        """
        self._data = np.load(path, allow_pickle=True)
        self._preproc_fn = preproc_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return process_rec(idx, self._data, self._preproc_fn)


class BaseDatasetTorch(Dataset):
    """
    This dataset subclass is used for reading data records
    and passing it to corresponding models in specified shape
    """

    def __init__(self, path, preproc_fn: Callable = None, aug_fn: Callable = None):
        """
        Constructor
        :param path: path to dataset file (.npy)
        :param preproc_fn: preprocessing function. Should return new data and new shape. (optional)
        :param aug_fn: augmentation function.
        """
        self._data = torch.load(path)
        self._preproc_fn = preproc_fn
        self._aug_fn = aug_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return process_torch(idx, self._data, self._preproc_fn, self._aug_fn)

    def pos_items(self):
        ps = 0
        for rcd in self._data:
            if rcd['label']:
                ps += 1
        return ps

    def neg_items(self):
        ng = 0
        for rcd in self._data:
            if not rcd['label']:
                ng += 1
        return ng
