#!/usr/bin/env
import argparse
import warnings
import os
import librosa

from core.pipeline import prepare_config
from tools.detector.detector_api import load_model, detect_cough
from tools.detector.utils import save_cough_series


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


def eval_track(config_path, model_path):
    _, model_conf, _, _ = prepare_config(config_path)

    # load model
    model = load_model(model_path,
                       model_conf.get_string('name'),
                       model_conf.get_list('args.input_shape')
                       )

    for root, _, files in os.walk(args.input):
        for name in files:
            path = os.path.join(root, name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    track, sr = librosa.load(path, sr=44100)
                    series = detect_cough(model, track, sr)
                    save_cough_series(series, path, args.output)
                except FileNotFoundError:
                    print(f'Fail to load: {path}')
                    continue


if __name__ == '__main__':
    args = parse_args()
    eval_track(args.config, args.model)
