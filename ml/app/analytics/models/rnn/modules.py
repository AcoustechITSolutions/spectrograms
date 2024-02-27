#!/usr/bin/env
from abc import ABC
import torch.nn as nn


""" Basic NN models building blocks """


class CNNEncoderBlock(nn.Module, ABC):

    def __init__(self, in_channels, out_channels, conv_dimension, dropout_pb=0.5):
        super(CNNEncoderBlock, self).__init__()

        if conv_dimension not in ['1d', '2d']:
            raise RuntimeError('Unsupported conv type')

        conv_fn = nn.Conv1d if conv_dimension == '1d' else nn.Conv2d
        batch_norm_fn = nn.BatchNorm1d if conv_dimension == '1d' else nn.BatchNorm2d

        self._conv_block = nn.Sequential(
            conv_fn(in_channels, out_channels, kernel_size=3),
            nn.LeakyReLU(),
            nn.Dropout(dropout_pb),
            batch_norm_fn(out_channels),
        ).apply(weights_init)

    def forward(self, batch):
        return self._conv_block(batch)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 0.0, 0.7)
        nn.init.normal_(m.bias, 0.0, 0.01)
