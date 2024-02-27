#!/usr/bin/env
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np
from models.sequential.base_sequential import BaseSequential
from models.modules import Attention
from models.cnn.base_cnn import BaseCNN
from models.cnn.ts_cnn10 import TSCNN
from models.algorithms import algorithms


class BaseAttentionCNN(nn.Module):

    def __init__(self, encoder, input_shape, num_classes, dropout_rate=.0):
        """
        Constructor
        :param encoder: torch module with cnn encoder to extract features
        :param input_shape: int or tuple with single item shape
        :param dropout_rate: float in range [0:1) with dropout rate
        """
        super(BaseAttentionCNN, self).__init__()
        self._shape = input_shape
        self._encoder = encoder
        self._num_classes = num_classes
        encoder_out_size = int(np.prod(self._encoder.out_dims))
        self._attention = Attention(encoder_out_size)

        # model head
        self._logits = nn.Sequential(
            nn.Linear(encoder_out_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    @abstractmethod
    def forward(self, batch):

        if isinstance(batch, tuple):
            assert len(batch) == 2
            batch, lengths = batch
        else:
            lengths = torch.tensor([batch[0].size(0)])

        time_steps, batch_size, h, w = batch[0].size()
        # run cnn feature extractor
        enc_in = batch[0].view(time_steps*batch_size, 1, h, w)
        enc_out = self._encoder.forward(enc_in)
        enc_out = enc_out.view(time_steps, batch_size, -1)

        attention_values = []
        for idx in range(enc_out.shape[1]):
            output = enc_out[:, idx, :]
            attention_val = self._attention(output[:int(lengths[0][idx]), :])
            attention_values.append(attention_val)
        attention_values = torch.stack(attention_values)
        if self._num_classes == 1:
            return torch.sigmoid(self._logits(attention_values))
        else:
            return self._logits(attention_values)


@algorithms.register('attention_cnn')
class AttentionCNN(BaseSequential):

    name = 'attention_cnn'

    def __init__(self, input_shape, num_classes, dropout_rate=.0):
        super(AttentionCNN, self).__init__(input_shape)

        self._net = BaseAttentionCNN(
            BaseCNN(input_shape, dropout_rate, ret_features=True),
            input_shape,
            num_classes,
            dropout_rate
        )

    @abstractmethod
    def forward(self, batch):
        return self._net(batch)


@algorithms.register('attention_tscnn')
class AttentionTSCNN(BaseSequential):

    name = 'attention_tscnn'

    def __init__(self, input_shape, num_classes, dropout_rate=.0):
        super(AttentionTSCNN, self).__init__(input_shape)
        self._shape = input_shape
        self._net = BaseAttentionCNN(
            TSCNN(input_shape, dropout_rate, ret_features=True),
            input_shape,
            num_classes,
            dropout_rate
        )

    @abstractmethod
    def forward(self, batch):
        return self._net(batch)

    @property
    def shape(self) -> tuple:
        return self._shape
