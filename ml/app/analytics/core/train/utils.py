#!/usr/bin/env
from collections import OrderedDict

import torch
import torch.nn as nn


def custom_load_dict(model, model_weights):

    new = list(model_weights.items())
    count = 0
    model_weights = OrderedDict()
    for key, value in model.state_dict().items():
        layer_name, weig = new[count]
        model_weights[key] = weig
        count += 1
    model.load_state_dict(model_weights)
    return model


class WeightedBCE(nn.Module):
    def __init__(self, weights_ar):

        super().__init__()
        self._weights = weights_ar

    def forward(self, output_, target_):
        weights_arr = self._weights
        if weights_arr is not None:
            assert len(weights_arr) == 2
            output_ = torch.clamp(output_, min=1e-8, max=1 - 1e-8)
            output_ = torch.where(torch.isnan(output_), torch.zeros_like(output_), output_)
            loss_v = weights_arr[1] * (target_ * torch.log(output_)) + \
                     weights_arr[0] * ((1 - target_) * torch.log(1 - output_))
        else:
            loss_v = target_ * torch.log(output_) + (1 - target_) * torch.log(1 - output_)

        return torch.neg(torch.mean(loss_v))