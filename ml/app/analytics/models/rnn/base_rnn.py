from abc import abstractmethod
import torch.nn as nn


class BaseRNN(nn.Module):
    """
    Basic interface for all RNN models
    """
    def __init__(self, in_shape):
        super(BaseRNN, self).__init__()
        self._shape = in_shape

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        return self._shape
