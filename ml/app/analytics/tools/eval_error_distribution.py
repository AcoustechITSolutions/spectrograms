#!/usr/bin/env
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from pathlib import Path

from core.train.eval import evaluate
from core.pipeline import prepare_config
from models.algorithms import algorithms
from metrics.confusion_metric import ConfusionMetric
from data.data_pipeline.dataset import BaseDataset
from data.data_pipeline.utils import make_collate_fn
from data.utils import generate_classification_metric, generate_detection_metric
from core.train.utils import custom_load_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to convert data into .nyp format')

    parser.add_argument('-m', '--model', type=str,
                        required=True, help='path to model')
    parser.add_argument('-c', '--config', type=str,
                        required=True, help='path to config file')
    parser.add_argument('-d', '--download-audio-path', type=str, required=False, default=None,
                        help='download path audio where the neural network made mistake')
    parser.add_argument('-s', '--source-table', action='store_true',
                        required=False, default=False,
                        help='enable source metric table')
    parser.add_argument('-md', '--metadata-table', action='store_true',
                        required=False, default=False,
                        help='enable metadata table')
    parser.add_argument('-t', "--confusion-types", nargs="+", default=["fp", "fn"],
                        help='types of confusion')

    return parser.parse_args()


def eval_func(config_path, model_path):
    data_conf, model_conf, _, train_conf = prepare_config(config_path)

    # load model
    model = algorithms.get(model_conf.get_string('name'))(**model_conf['args'])
    model_weights = torch.load(Path(model_path))
    model = custom_load_dict(model, model_weights)
    # init CPU device
    device = torch.device('cpu')
    # load model into CPU
    model = model.to(device)
    # init data loader
    dataset = BaseDataset(data_conf.get_string('generated.eval_data'))
    dataloader = DataLoader(
        dataset,
        train_conf.get_int('batch_size'),
        collate_fn=make_collate_fn(device, model)
    )

    records = np.load(data_conf.get_string('generated.eval_data'), allow_pickle=True)
    source = [rec['source'] for rec in records]
    cms = evaluate(model, dataloader, ConfusionMetric())

    if data_conf.get_string('name') in 'detection':
        print(generate_detection_metric(source, cms, args.confusion_types,))
    else:
        if args.source_table:
            print('Source metric:')
            print(generate_classification_metric(data_conf, source, cms))
        if args.metadata_table:
            print('Metadata:')
            print(generate_classification_metric(data_conf,
                                                 source,
                                                 cms,
                                                 args.confusion_types,
                                                 args.download_audio_path
                                                 ))


if __name__ == '__main__':
    args = parse_args()
    eval_func(args.config, args.model)
