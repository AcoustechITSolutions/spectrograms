#!/usr/bin/env
from abc import abstractmethod

import torch.nn as nn
from models.cnn.modules import TSAttention

from models.modules import weights_init
from models.algorithms import algorithms


@algorithms.register('tscnn_10')
class TSCNN(nn.Module):

    name = 'tscnn_10'

    def __init__(self, shape, dropout_pb=.0, ret_features=False):
        """
        Constructor
        :param shape: tuple with input shape
        :param dropout_pb: dropout rate
        :param ret_features: flag to return CNN features instead probability
        """
        super(TSCNN, self).__init__()

        def make_block(ch, height, width):
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode='circular'),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.Dropout(dropout_pb),
                nn.Conv2d(ch, 2 * ch, kernel_size=3, padding=1, padding_mode='circular'),
                nn.BatchNorm2d(2 * ch),
                nn.ReLU(),
                nn.Dropout(dropout_pb),
                nn.AvgPool2d(2, 2),
                TSAttention((2 * ch,  height // 2, width // 2))
            ).apply(weights_init)

        channels_num = 16
        self._init_conv = nn.Conv2d(
            1,
            channels_num,
            kernel_size=7,
            padding=3,
            padding_mode='circular'
        )

        blocks = []
        h, w = shape
        while channels_num < 256:
            blocks.append(make_block(channels_num, h, w))
            h, w = h // 2, w // 2
            channels_num *= 2
        self._blocks = nn.Sequential(*blocks)

        self._head = nn.Sequential(
            nn.Conv2d(channels_num // 2, 1, kernel_size=3, padding=1, padding_mode='circular'),
            nn.AvgPool2d(kernel_size=shape),
            nn.Sigmoid()
        )

        self._ret_features = ret_features
        self._out_dims = h, w, channels_num if ret_features else 1

    @abstractmethod
    def forward(self, batch):
        # compute features
        batch = self._init_conv(batch)
        features = self._blocks(batch)
        if self._ret_features:
            return features
        return self._head(features)

    @property
    def out_dims(self):
        return self._out_dims
