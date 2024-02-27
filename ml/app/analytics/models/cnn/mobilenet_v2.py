#!/usr/bin/env
from abc import abstractmethod
import torch.nn as nn
import torchvision.models as models

from models.algorithms import algorithms


@algorithms.register('audio_mobilenet')
class AudioMobileNet(nn.Module):
    """
    Modified for audio recognition pretrained on ImageNet MobileNet v2 model
    """
    def __init__(self):
        super(AudioMobileNet, self).__init__()
        # this stage is needed for adding channels to spectrogram
        # (mobilenet has 3 input channels)
        self._prepare_stage = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self._backbone = models.mobilenet_v2(pretrained=True)
        # replace mobilenet linear layer with custom layer
        self._backbone.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    @abstractmethod
    def forward(self, batch):
        batch = self._prepare_stage(batch)
        return self._backbone(batch)
