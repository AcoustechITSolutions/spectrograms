#!/usr/bin/env
import os
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
import librosa
import librosa.feature
import librosa.display
import numpy as np
import pylab
import cv2

from data.data_pipeline.utils import make_collate_fn
from core.pipeline import prepare_config
from tools.noiser.noiser import DatasetWithoutLoad
from tools.load_model import load_model
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to evaluate one record using the model')

    parser.add_argument('-m', '--model', type=str,
                        required=True, help='path to model')
    parser.add_argument('-c', '--config', type=str,
                        required=True, help='path to config file')
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='input path')
    parser.add_argument('-o', '--output', type=str,
                        required=True, help='output path')

    return parser.parse_args()


def visual(path_input, model, path_output):

    seq_elem_size = model.shape[-1]
    if not os.path.exists(os.path.join(path_output, 'spectrogram')):
        os.mkdir(os.path.join(path_output, 'spectrogram'))
    folder = os.path.join(path_output, 'spectrogram')
    track, sr = librosa.load(path_input, sr=44100)

    data = [{
        'label': 1,
        'data': [librosa.feature.mfcc(track)],
        'source': path_input,
    }]
    print(f'TRACK SHAPE: {librosa.feature.mfcc(track).shape}')

    preds, weg = get_pred(data, model)

    track_mel = librosa.feature.melspectrogram(track)
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge

    librosa.display.specshow(librosa.power_to_db(track_mel, ref=np.max))
    pylab.savefig(folder + '/raw_spec.jpg', bbox_inches=None, pad_inches=0)
    pylab.close()
    for i in range((track_mel.shape[1] // seq_elem_size) + 1):
        if track_mel.shape[1] - i * seq_elem_size < seq_elem_size:
            w = track_mel.shape[0]
            h = track_mel.shape[1] - i * seq_elem_size
            if track_mel.shape[1] - i * seq_elem_size < (seq_elem_size / 2) + 1:
                print(f'sequence #{i+1} with size {w, h} sequence length < {seq_elem_size} / 2, not evaluated')
            else:
                print(f'sequence #{i + 1} with size {w, h} and attention weight {weg[i]}')
        else:
            w = track_mel.shape[0]
            h = seq_elem_size
            print(f'sequence #{i+1} with size {w, h} and attention weight {weg[i]}')
        audio = track_mel[:, seq_elem_size * i:(seq_elem_size * i) + h]
        audio = librosa.feature.inverse.mel_to_audio(audio)
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        S = librosa.feature.melspectrogram(y=audio, sr=44100)
        if h != seq_elem_size:
            S = np.pad(S, (0,seq_elem_size-S.shape[1]))[:S.shape[0]]

        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(folder + f'/test{i}.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        S = cv2.imread(folder + f'/test{i}.jpg')
        if h != seq_elem_size:
            S = S[:, :int(S.shape[1] * h/seq_elem_size)]
        os.remove(folder + f'/test{i}.jpg')
        if h > seq_elem_size / 2:
            S = S * float(weg[i] + 1.05 - max(weg))
        else:
            S = S * float(min(weg) + 1.05 - max(weg))
        if i == 0:
            full = S
            width = S.shape[0]
        else:
            S = cv2.resize(S, (int(S.shape[1] * (S.shape[0] / width)), width))
            full = cv2.hconcat([full, S])
    raw = cv2.imread(folder + '/raw_spec.jpg')
    os.remove(folder + '/raw_spec.jpg')
    raw = cv2.resize(raw, (full.shape[1], full.shape[0]))

    cv2.imwrite(folder + '/raw_spec.jpg', raw)
    cv2.imwrite(folder + '/weighted_spec.jpg', full)
    diag = 'COVID' if preds.item() > 0.5 else 'NO COVID'
    print(f'MODEL PREDICTION: {round(float(preds.item()), 5)}  ::: {diag} \n'
          f'Attention weights: {weg}')


def get_pred(data, model):

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

    weg = model._net._attention.attention_weights.tolist()

    return preds, weg


def prepare(config_path, model_path):
    _, model_conf, _, _ = prepare_config(config_path)

    # load model
    model = load_model(model_path,
                       model_conf.get_string('name'),
                       model_conf.get_list('args.input_shape'),
                       model_conf['args.dropout_rate']
                       )

    return model

if __name__ == '__main__':
    args = parse_args()
    model = prepare(args.config, args.model)
    visual(args.input, model, args.output)