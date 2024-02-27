#!/usr/bin/env
import argparse
import warnings
warnings.filterwarnings("ignore")
import librosa
import torch
from torch.utils.data import DataLoader
from pathlib import PurePath

from data.data_pipeline.utils import make_collate_fn
from core.pipeline import prepare_config
from tools.noiser.noiser import DatasetWithoutLoad
from tools.load_model import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to evaluate one record using the model')

    parser.add_argument('-m', '--model', type=str,
                        required=True, help='path to model')
    parser.add_argument('-c', '--config', type=str,
                        required=True, help='path to config file')
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='input path')

    return parser.parse_args()


def prepare(config_path, model_path, input_path):
    _, model_conf, _, _ = prepare_config(config_path)

    # load model
    model = load_model(model_path,
                       model_conf.get_string('name'),
                       model_conf.get_list('args.input_shape'),
                       model_conf['args.dropout_rate']
                       )
    track, _ = librosa.load(input_path, sr=44100)
    data = [{
        'label': 1,
        'data': [librosa.feature.mfcc(track)],
        'source': input_path,
    }]
    return model, data


def make_pred(data, model):
    dataset = DatasetWithoutLoad(data=data)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn=make_collate_fn(torch.device('cuda'),
                                                       model))

    model.eval()

    for idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            _, data, pad_lengths = batch
            preds = model.forward((data, pad_lengths))
        else:
            _, data = batch
            preds = model.forward(data)

    return preds


if __name__ == '__main__':
    args = parse_args()
    model, data = prepare(args.config, args.model, args.input)
    pred = make_pred(data, model)
    print(f'Probability this audio ({PurePath(args.input).parts[-1]}) is noisy: {round(float(pred.item()), 5)}')
