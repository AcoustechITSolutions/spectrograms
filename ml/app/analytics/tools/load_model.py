#!/usr/bin/env
import torch
from pathlib import Path

from models.algorithms import algorithms


def load_model(model_path, model_name='attention_cnn', shape=(20, 256), dropout=0.1):
    model = algorithms.get(model_name)(shape, dropout)
    device = torch.device('cuda')
    model = model.to(device)
    model.load_state_dict(torch.load(Path(model_path)))
    return model