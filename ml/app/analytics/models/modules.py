#!/usr/bin/env
from abc import abstractmethod

import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, size):
        super(Attention, self).__init__()
        weights = torch.zeros([1, size], dtype=torch.float32, requires_grad=True)
        self._weights = nn.Parameter(nn.init.xavier_normal_(weights).squeeze(0))
        self._linear = nn.Linear(size, size)

    @abstractmethod
    def forward(self, sequence):
        # compute weights scores
        activations = torch.tanh(self._linear(sequence))
        scores = torch.matmul(activations, self._weights)
        self._attention_weights = torch.softmax(scores, dim=0)
        return torch.matmul(self._attention_weights, sequence)

    @property
    def attention_weights(self):
        return self._attention_weights


class CNNBlock(nn.Module):
    """
    Basic block for cnn model
    """
    def __init__(self, in_ch, out_ch, h, w, dropout_pb=.0):
        super(CNNBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.MaxPool2d(kernel_size=2)
        ).apply(weights_init)

    @abstractmethod
    def forward(self, batch):
        return self._block(batch)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
