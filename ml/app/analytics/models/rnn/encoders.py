#!/usr/bin/env
from abc import abstractmethod
import torch.nn as nn

from models.rnn.modules import CNNEncoderBlock


class Encoder(nn.Module):
    """
    Base class for feature extraction
    """
    def __init__(self, dropout_pb=.15):
        """
        Constructor
        :param dropout_pb: dropout probability rate
        """
        super(Encoder, self).__init__()
        self._dropout_pb = dropout_pb

    @abstractmethod
    def forward(self, batch):
        raise NotImplemented

    @abstractmethod
    def output_shape(self, shape):
        """
        Compute encoder output tensor shape due to specified input shape
        :param shape: shape of feature extractor input tensor 
        """
        raise NotImplemented


class CNNEncoder1d(Encoder):

    def __init__(self, dropout_pb):
        """
        Constructor
        :param dropout_pb: dropout rate
        """
        super(CNNEncoder1d, self).__init__()
        self._out_channels = 32
        max_pool_ker = 3

        def cnn_encoder_block_1d(in_ch, out_ch):
            return nn.Sequential(
                CNNEncoderBlock(1, 4, '1d', dropout_pb),
                nn.MaxPool1d(kernel_size=max_pool_ker)
            )

        self._cnn_encoder = nn.Sequential(
            cnn_encoder_block_1d(1, 4),
            cnn_encoder_block_1d(4, 8),
            cnn_encoder_block_1d(8, 16),
            cnn_encoder_block_1d(16, self._out_channels),
        )

    @abstractmethod
    def forward(self, batch):
        return self._cnn_encoder(batch)

    @abstractmethod
    def output_shape(self, shape):
        for _ in range(len(self._cnn_encoder)):
            # dim squeeze after conv with kernel 3
            shape = shape - 2
            # dim squeeze after max polling with kernel 3
            shape = int((shape - shape % 3) / 3)
        return self._out_channels, shape


class CNNEncoder2d(Encoder):

    def __init__(self, dropout_pb):
        """
        Constructor
        :param dropout_pb: dropout rate
        """
        super(CNNEncoder2d, self).__init__()
        self._out_channels = 32
        self._cnn_encoder = nn.Sequential(
            CNNEncoderBlock(1, 8, '2d', dropout_pb),
            CNNEncoderBlock(8, 16, '2d', dropout_pb),
            CNNEncoderBlock(16, self._out_channels, '2d', dropout_pb),
            CNNEncoderBlock(self._out_channels, self._out_channels, '2d', dropout_pb),
        )

    @abstractmethod
    def forward(self, batch):
        return self._cnn_encoder(batch)

    @abstractmethod
    def output_shape(self, shape):
        h, w = shape[-2:]
        dim_diff = 2 * len(self._cnn_encoder)
        return self._out_channels, h - dim_diff, w - dim_diff
