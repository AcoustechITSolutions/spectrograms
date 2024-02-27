#!/usr/bin/env
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np

from models.algorithms import algorithms
from models.rnn.encoders import Encoder
from models.rnn.base_rnn import BaseRNN
from models.rnn.encoders import CNNEncoder2d, CNNEncoder1d
from models.modules import Attention


class RCNN(BaseRNN):
    """
    class for RNNs nets.
    """
    def __init__(self, in_shape, encoder: Encoder, attention=True):
        """
        Constructor
        :param in_shape: int or tuple with single item input shape
        :param encoder: instance of Encoder subclass as feature extractor for LSTM module
        :param attention: flag to use attention block
        """
        super(RCNN, self).__init__(in_shape)
        # define cnn based encoder
        max_pool_ker, dropout_pb = 3, 0.7
        self._encoder = encoder

        lstm_hidden_units_num = 512
        self._lstm = nn.LSTM(
            np.prod(encoder.output_shape(in_shape)),
            lstm_hidden_units_num,
            1
        )

        # model head
        self._logits = nn.Sequential(
            nn.Linear(lstm_hidden_units_num, 256),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.Linear(256, 1),
        )

        self._attention = Attention(lstm_hidden_units_num) if attention else None

    @abstractmethod
    def forward(self, batch):
        if isinstance(batch, tuple):
            assert len(batch) == 2
            batch, lengths = batch
        else:
            lengths = torch.ones(batch.size()[1]) * (-1)
        time_steps, batch_size, *_ = batch.size()
        # run cnn feature extractor
        enc_out = []
        for ts in range(time_steps):
            enc_out.append(
                self._encoder.forward(
                    torch.unsqueeze(batch[ts], 1)
                )
            )
        enc_out = torch.cat(enc_out)
        # flatten tensor and reshape it for rnn
        enc_out = enc_out.view(time_steps, batch_size, -1)
        # run rnn
        rnn_out, _ = self._lstm(enc_out)

        if self._attention:
            rnn_outs = []
            for idx in range(rnn_out.shape[1]):
                output = rnn_out[:, idx, :]
                output = self._attention(output[:int(lengths[idx]), :])
                rnn_outs.append(output)
            return torch.sigmoid(self._logits(torch.stack(rnn_outs)))

        # get last valued lstm output
        rnn_last_out = []
        for idx in range(rnn_out.shape[1]):
            output = rnn_out[:, idx, :]
            rnn_last_out.append(output[int(lengths[idx]) - 1].unsqueeze(0))
        rnn_last_out = torch.cat(rnn_last_out)
        return torch.sigmoid(self._logits(rnn_last_out))


@algorithms.register('rcnn_2d')
class RCNN2d(BaseRNN):

    name = 'rcnn_2d'

    def __init__(self, **args):
        super(RCNN2d, self).__init__(args['input_shape'])
  
        self._rcnn = RCNN(args['input_shape'], CNNEncoder2d(args['dropout_rate']))

    @abstractmethod
    def forward(self, batch):
        return self._rcnn(batch)


@algorithms.register('rcnn_1d')
class RCNN1d(BaseRNN):

    name = 'rcnn_1d'

    def __init__(self, **args):
        super(RCNN1d, self).__init__(args['input_shape'])
        self._rcnn = RCNN(args['input_shape'], CNNEncoder1d(args['dropout_rate']))

    @abstractmethod
    def forward(self, batch):
        return self._rcnn(batch)
