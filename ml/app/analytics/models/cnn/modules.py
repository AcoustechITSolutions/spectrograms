#!/usr/bin/env
from abc import abstractmethod
import torch
import torch.nn as nn

from models.modules import weights_init


class TSAttention(nn.Module):

    def __init__(self, shape):
        super(TSAttention, self).__init__()

        c, h, w = shape

        self._weights = nn.Parameter(torch.Tensor([1/3, 1/3, 1/3]), requires_grad=True)

        self._freq_w = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, padding_mode='circular'),
            nn.AvgPool2d(kernel_size=(1, w))
        ).apply(weights_init)

        self._time_w = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, padding_mode='circular'),
            nn.AvgPool2d(kernel_size=(h, 1))
        ).apply(weights_init)

    @abstractmethod
    def forward(self, batch):
        freq_features = self._freq_w(batch) * batch
        time_features = self._time_w(batch) * batch
        a, b, c = torch.softmax(self._weights, dim=0)
        return a * batch + b * freq_features + c * time_features
