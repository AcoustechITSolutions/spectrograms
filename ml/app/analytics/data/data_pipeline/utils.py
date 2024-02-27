#!/usr/bin/env
import torch
import torch.nn.functional as f
from typing import Callable, Sequence
import numpy as np
import cv2 as cv

from models.rnn.base_rnn import BaseRNN
from models.sequential.base_sequential import BaseSequential


def make_sequential_collate_fn(device, seq_elem_size):
    """
    Factory function to create collate function for sequential models
    :param device: PyTorch device entity. This is context for CPU/GPU software environment
    :param seq_elem_size: size of sequential element
    :return collate function
    """

    def collate_fn(batch):
        """
        collate function is necessary for processing variable length tensors.
        It pads missed parts and return padding mask (1 for real data and 0 for padded item).
        Also this function passes all tensors to device
        :param batch: Input tensor with labels and data samples
        :return tuple with labels and batch tensors, padding mask tensor and padding lengths for sequence models
        """

        # repack labels and batch into separate entities
        labels = torch.tensor([e[0] for e in batch], dtype=torch.float32).to(device)

        proc_batch = []
        for _, count in batch:
            proc = []
            for data in count:
                dim = data[0].shape
                seq_len = max(dim[-1] // seq_elem_size, 1)
                if len(dim) == 2:
                    _, w = dim
                    # pad too small spectrogram
                    if w < seq_elem_size:
                        data = f.pad(
                            input=data,
                            pad=[0, seq_elem_size - w, 0, 0],
                            mode='constant',
                            value=0
                        )
                    elif w % seq_elem_size > (seq_elem_size * 0.5):
                        data = f.pad(
                            input=data,
                            pad=[0, seq_elem_size - (w % seq_elem_size), 0, 0],
                            mode='constant',
                            value=0
                        )
                        seq_len += 1

                    data = data[:, :, 0:seq_len * seq_elem_size]

                data = torch.split(data, seq_elem_size, dim=-1)
                data = torch.stack(data)
                proc.append(torch.squeeze(data, dim=1))
            proc_batch.append(proc)
        reshaped_batch = [[], [], []]
        for item in range(len(proc_batch)):
            for n in range(len(proc_batch[item])):
                reshaped_batch[n].append(proc_batch[item][n])
        proc_batch = reshaped_batch[:n + 1]

        # get sequence lengths
        lengths = []
        for item in proc_batch:
            lengths.append([part.shape[0] for part in item])
        # pad
        for item in range(len(proc_batch)):
            proc_batch[item] = torch.nn.utils.rnn.pad_sequence(proc_batch[item]).to(device)

        for item in range(len(lengths)):
            lengths[item] = torch.tensor(lengths[item], dtype=torch.long).to(device)

        return labels, proc_batch, lengths

    return collate_fn


def make_collate_fn(device, model):
    """
    Factory function for creating collate function based on model
    :param device: torch device entity to transfer data on
    :param model: torch model
    :return collate function
    """
    if issubclass(type(model), (BaseRNN, BaseSequential)):
        return make_sequential_collate_fn(device, model.shape[-1])
    return make_simple_collate_fn(device)


def make_simple_collate_fn(device):
    """
    Factory function to create collate function for pytorch dataloader in case of non recurrent model
    :param device: PyTorch device entity. This is context for CPU/GPU software environment
    :return collate function
    """

    def collate_fn(batch):
        """
        collate function is necessary for transferring data into GPU
        :param batch: Input tensor
        :return tuple with labels and batch tensors
        """
        labels = torch.tensor([e[0] for e in batch], dtype=torch.float32).to(device)
        batch = torch.stack([e[1] for e in batch]).to(device)
        return labels, batch

    return collate_fn


def make_torch_tensor(collection):
    """
    Helper function to make torch torch tensor
    :param collection: data to make torch tensor from
    :return torch tensor
    """
    if not isinstance(collection, np.ndarray):
        collection = np.array(collection, dtype=np.float32)
    return torch.from_numpy(collection).type('torch.FloatTensor')


def clip_and_pad_edit(data, target_shape):
    """
    Preprocessing function for resizing 2d image by clipping and padding.
    Function clips too big image and pad it with zero if an image to small.
    :param data: np array input image
    :param target_shape: shape to resize the image
    :return numpy array with resized image
    """
    i_h, i_w = data.shape
    t_h, t_w = target_shape
    # clip image
    data = data[0:min(t_h, i_h), 0:min(i_w, t_w)]
    if data.shape == target_shape:
        return data
    # pad image
    padding = (0, t_h - i_h), (0, t_w - i_w)
    return np.pad(data, padding, 'constant')


def resize(data, target_shape):
    """
    Preprocessing function for resizing 2d image by interpolation.
    :param data: np array input image
    :param target_shape: shape to resize the image
    :return numpy array with resized image
    """
    return cv.resize(data, dsize=target_shape[::-1], interpolation=cv.INTER_AREA)


def process_rec(idx, recs: Sequence, preproc_fn: Callable = None):
    """
    Preprocess loaded record and converts it to torch tensor
    :param idx: index of preprocessed record
    :param recs: sequence of dicts with sample info
    :param preproc_fn: preprocessing function (optional)
    :return 2 torch tensors. The first one with label and the second one with data
    """
    rec = dict(recs[idx])
    batch = []
    for i in range(len(rec['data'])):
        batch.append(make_torch_tensor((rec['data'][i])).unsqueeze(dim=0))

    return make_torch_tensor(rec['label']), batch


def process_torch(idx, recs: Sequence, preproc_fn: Callable = None, aug_fn: Callable = None):
    """
    Preprocess loaded record and converts it to torch tensor
    :param idx: index of preprocessed record
    :param recs: sequence of dicts with sample info
    :param preproc_fn: preprocessing function (optional)
    :param aug_fn: augmentation function (optional)
    :return: 2 torch tensors. The first one with label and the second one with data
    """
    rec = dict(recs[idx])
    batch = rec['data']
    if aug_fn is not None:
        batch = aug_fn(batch.unsqueeze(0))[0]
    batch = preproc_fn(batch)

    return make_torch_tensor(rec['label']), batch


def make_torch_collate_fn(device, seq_elem_size):
    """
    Factory function to create collate function for sequential models
    :param device: PyTorch device entity. This is context for CPU/GPU software environment
    :param seq_elem_size: size of sequential element
    :return collate function
    """

    def collate_fn(batch):
        """
        collate function is necessary for processing variable length tensors.
        It pads missed parts and return padding mask (1 for real data and 0 for padded item).
        Also this function passes all tensors to device
        :param batch: Input tensor with labels and data samples
        :return tuple with labels and batch tensors, padding mask tensor and padding lengths for sequence models
        """

        # repack labels and batch into separate entities
        labels = torch.tensor([e[0] for e in batch], dtype=torch.float32).to(device)

        proc_batch = []
        for _, data in batch:
            proc = []

            dim = data.size()
            seq_len = max(dim[-1] // seq_elem_size, 1)
            if len(dim) == 3:
                _, _, w = dim
                # pad too small spectrogram
                if w < seq_elem_size:
                    data = f.pad(
                        input=data,
                        pad=[0, seq_elem_size - w, 0, 0],
                        mode='constant',
                        value=0
                    )
                elif w % seq_elem_size > (seq_elem_size * 0.5):
                    data = f.pad(
                        input=data,
                        pad=[0, seq_elem_size - (w % seq_elem_size), 0, 0],
                        mode='constant',
                        value=0
                    )
                    seq_len += 1

                data = data[:, :, 0:seq_len * seq_elem_size]

            data = torch.split(data, seq_elem_size, dim=-1)
            data = torch.stack(data)
            proc.append(torch.squeeze(data, dim=1))
            proc_batch.append(proc)
        reshaped_batch = [[], [], []]
        for item in range(len(proc_batch)):
            for n in range(len(proc_batch[item])):
                reshaped_batch[n].append(proc_batch[item][n])
        proc_batch = reshaped_batch[:n + 1]

        # get sequence lengths
        lengths = []
        for item in proc_batch:
            lengths.append([part.shape[0] for part in item])
        # pad
        for item in range(len(proc_batch)):
            proc_batch[item] = torch.nn.utils.rnn.pad_sequence(proc_batch[item]).to(device)
        for item in range(len(lengths)):
            lengths[item] = torch.tensor(lengths[item], dtype=torch.long).to(device)

        return labels, proc_batch, lengths

    return collate_fn
