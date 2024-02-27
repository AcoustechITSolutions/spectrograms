#!/usr/bin/env
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Callable
import argparse
from pathlib import Path
import warnings
import librosa

from core.data_format import Record
from core.pipeline import prepare_config
from data.data_pipeline.utils import make_collate_fn, process_rec
from models.algorithms import algorithms

"""
    This script evaluates one record using the model
"""


class NewBaseDataset(Dataset):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to evaluate one record using the model')

    parser.add_argument('-m', '--model', type=str,
                        required=True, help='path to model')
    parser.add_argument('-c', '--config', type=str,
                        required=True, help='path to config file')
    parser.add_argument('-t', '--track', type=str,
                        required=True, help='path to audio file')
    return parser.parse_args()


def eval_track(config_path, model_path, track_path):

    _, model_conf, _, _ = prepare_config(config_path)

    recs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            track, sr = librosa.load(Path(track_path), sr=44100)
        except OSError:
            print('File ', track_path, 'does not exist')
            return

    track = librosa.feature.melspectrogram(track, sr)
    recs.append(Record(0.0, sr, track.shape, track.flatten()))

    # load model
    model = algorithms.get(model_conf.get_string('name'))(**model_conf['args'])
    model.load_state_dict(torch.load(Path(model_path)))

    # init data loader
    dataset = NewBaseDataset(data=recs)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=make_collate_fn(torch.device('cpu'), model)
    )

    model.eval()
    label, data, _, pad_lengths = next(iter(dataloader))
    preds = model.forward((data, pad_lengths))
    diagnosis = 'covid' if torch.round(preds) == 1.0 else 'healthy'
    print(f'Evaluation finished.\n Prediction: {preds.item()}\n'
          f' Model diagnosis: {diagnosis}')


if __name__ == '__main__':
    args = parse_args()
    eval_track(args.config, args.model, args.track)
