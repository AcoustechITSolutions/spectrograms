#!/usr/bin/env
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import os
from typing import Callable

from data.data_pipeline.utils import make_collate_fn, process_rec


def make_records_for_detector(track, sr, window_step, signal_duration):
    sample_window_step = window_step * sr
    sample_signal_dur = signal_duration * sr

    start_samples = [int(sample) for sample in np.arange(0, len(track), sample_window_step)
                     if sample + sample_signal_dur < len(track)]

    tracks = []
    for start in start_samples:
        data = librosa.feature.mfcc(track[start:start + sample_signal_dur])
        tracks.append({'label': 1.0,
                       'data': [data],
                       'timing': [start, start + sample_signal_dur]
                       })
    return tracks


def make_cough_series(model, record, samplerate, signal_duration):
    series = []

    dataset = DatasetWithoutLoad(data=record)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=make_collate_fn(torch.device('cpu'), model))
    model.eval()

    start_sample, end_sample, counter = 0, 0, 0
    for idx, batch in enumerate(dataloader):
        preds = model.forward((batch[1], batch[2]))

        if torch.round(preds) == 1.0:
            if counter == 0:
                start_sample = record[idx]['timing'][0]
            counter += 1

            if idx == len(dataloader) - 1:
                series.append((start_sample, record[idx]['timing'][1]))
        elif torch.round(preds) == 0.0:
            if counter != 0:
                end_sample = record[idx - 1]['timing'][1]
                if end_sample - start_sample > samplerate * signal_duration:
                    series.append((start_sample, end_sample))
                counter = 0

    return series


def save_cough_series(series, file_path, output_path):
    if not series:
        return
    for idx, data in enumerate(series):
        start, end = data
        new_filename = f"{output_path}/{os.path.splitext(os.path.basename(file_path))[0]}_{idx}.wav"
        track, sr = librosa.load(file_path, sr=44100)
        librosa.output.write_wav(new_filename, track[int(start):int(end)], sr=44100)


class DatasetWithoutLoad(Dataset):
    """
    This dataset subclass is used for reading data records
    and passing it to corresponding models in specified shape
    """
    def __init__(self, data, preproc_fn: Callable = None):
        """
        Constructor
        :param data: path to dataset file (.npy)
        :param preproc_fn: preprocessing function. Should return new data and new shape. (optional)
        """
        self._data = data
        self._preproc_fn = preproc_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return process_rec(idx, self._data, self._preproc_fn)
