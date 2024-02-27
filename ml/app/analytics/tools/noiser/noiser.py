#!/usr/bin/env
from typing import Callable
from torch.utils.data import Dataset

from data.data_pipeline.utils import process_rec


class DatasetWithoutLoad(Dataset):
    """
    This dataset subclass is used for reading data records
    and passing it to corresponding models in specified shape
    """
    def __init__(self, data, preproc_fn: Callable = None):
        """
        Constructor
        :param data: path to dataset file (.npy)
        :param preproc_fn: preprocessing function. Should return new data and new shape. (optional)
        """
        self._data = data
        self._preproc_fn = preproc_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return process_rec(idx, self._data, self._preproc_fn)