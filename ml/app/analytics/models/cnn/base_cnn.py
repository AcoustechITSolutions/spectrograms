#!/usr/bin/env
from abc import abstractmethod
import torch.nn as nn
from models.modules import CNNBlock

from models.algorithms import algorithms


@algorithms.register('base_cnn')
class BaseCNN(nn.Module):

    def __init__(self, shape, dropout_pb=.0, ret_features=False):
        """
        Constructor
        :param shape: tuple with input shape
        :param dropout_pb: dropout rate
        :param ret_features: flag to return CNN features instead probability
        """
        super(BaseCNN, self).__init__()
        h, w = shape

        # first block
        channels_num = 32
        self._in_block = nn.Sequential(
            nn.Conv2d(1, channels_num, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.BatchNorm2d(channels_num),
            nn.MaxPool2d(kernel_size=2)
        )

        modules = []
        count = 1
        while channels_num < 256:
            modules.append(CNNBlock(channels_num,
                                    2 * channels_num,
                                    int(h * (0.5 ** count)),
                                    int(w * (0.5 ** count)),
                                    dropout_pb * (0.5 ** count)))
            channels_num *= 2
            count += 1

        self._main_block = nn.Sequential(*modules)

        dim_scale_factor = 2 ** (len(modules) + 1)

        if ret_features:
            self._head = None
            self._out_dims = (h // dim_scale_factor, w // dim_scale_factor, channels_num)
        else:
            self._out_dims = 1
            self._head = nn.Sequential(
                nn.Conv2d(channels_num, 1, kernel_size=3, padding=1, padding_mode='circular'),
                nn.AvgPool2d(kernel_size=(h // dim_scale_factor, w // dim_scale_factor)),
                nn.Sigmoid()
            )

    @abstractmethod
    def forward(self, batch):
        batch = self._in_block(batch)
        batch = self._main_block(batch)
        return self._head(batch) if self._head else batch

    @property
    def out_dims(self):
        return self._out_dims
