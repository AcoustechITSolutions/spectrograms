#!/usr/bin/env
import torch
from pathlib import Path

from tools.detector.utils import make_cough_series, make_records_for_detector
from models.algorithms import algorithms


WINDOW_STEP = 0.5
SIGNAL_DURATION = 1


def load_model(model_path, model_name='attention_tscnn', shape=(20, 87)):
    model = algorithms.get(model_name)(shape)
    model.load_state_dict(torch.load(Path(model_path)))
    return model


def detect_cough(model, track, samplerate):
    return make_cough_series(model,
                             make_records_for_detector(track, samplerate, WINDOW_STEP, SIGNAL_DURATION),
                             samplerate,
                             SIGNAL_DURATION)
