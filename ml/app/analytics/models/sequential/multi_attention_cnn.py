#!/usr/bin/env
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np

from models.sequential.base_sequential import BaseSequential
from models.modules import Attention
from models.cnn.base_cnn import BaseCNN
from models.algorithms import algorithms

class MultiAttentionCNN(nn.Module):

    def __init__(self, encoder_seq, input_shape, dropout_rate=.0):
        """
        Constructor
        :param encoder: torch module with cnn encoder to extract features
        :param input_shape: int or tuple with single item shape
        :param dropout_rate: float in range [0:1) with dropout rate
        """
        super(MultiAttentionCNN, self).__init__()
        self._shape = input_shape
        encoder_out_size = int(np.prod(encoder_seq[0].out_dims))

        for i in range(len(encoder_seq)):
            setattr(self, f'_encoder_{i}', encoder_seq[i])
            setattr(self, f'_attention_{i}', Attention(encoder_out_size))
        # model head
        self._logits = nn.Sequential(
            nn.Linear(encoder_out_size * len(encoder_seq), 256 * len(encoder_seq)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256 * len(encoder_seq), 1),
        )

    @abstractmethod
    def forward(self, batch):

        if isinstance(batch, tuple):
            assert len(batch) == 2
            data, lengths = batch
        else:
            data = batch
            lengths = []
            for item in batch:
                lengths.append(torch.tensor([item.size()[0]]))
        attention_out = []
        for i in range(len(data)):
            time_steps, batch_size, h, w = data[i].size()
            # run cnn feature extractor
            enc_in = data[i].view(time_steps*batch_size, 1, h, w)
            enc_out = getattr(self, f'_encoder_{i}').forward(enc_in)
            enc_out = enc_out.view(time_steps, batch_size, -1)

            attention_values = []
            for idx in range(enc_out.shape[1]):
                output = enc_out[:, idx, :]
                attention_val = getattr(self, f'_attention_{i}')(output[:int(lengths[i][idx]), :])
                attention_values.append(attention_val)
            attention_out.append(torch.stack(attention_values))

        return torch.sigmoid(self._logits(torch.cat(attention_out, dim=1)))

@algorithms.register('attention_cnn_double')
class Double_AttentionCNN(BaseSequential):

    name = 'attention_cnn_double'

    def __init__(self, input_shape, dropout_rate=.0):
        super(Double_AttentionCNN, self).__init__(input_shape)

        self._net = MultiAttentionCNN(
            [BaseCNN(input_shape, dropout_rate, ret_features=True),
            BaseCNN(input_shape, dropout_rate, ret_features=True)],
            input_shape,
            dropout_rate
        )

    @abstractmethod
    def forward(self, batch):
        return self._net(batch)

@algorithms.register('attention_cnn_triple')
class Triple_AttentionCNN(BaseSequential):

    name = 'attention_cnn_triple'

    def __init__(self, input_shape, dropout_rate=.0):
        super(Triple_AttentionCNN, self).__init__(input_shape)

        self._net = MultiAttentionCNN(
            [BaseCNN(input_shape, dropout_rate, ret_features=True),
            BaseCNN(input_shape, dropout_rate, ret_features=True),
            BaseCNN(input_shape, dropout_rate, ret_features=True)],
            input_shape,
            dropout_rate
        )

    @abstractmethod
    def forward(self, batch):
        return self._net(batch)