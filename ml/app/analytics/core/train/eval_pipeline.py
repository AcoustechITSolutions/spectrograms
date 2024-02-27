#!/usr/bin/env
import torch
from torch.utils.data import DataLoader
import torchaudio.transforms as T

from data.data_pipeline.dataset import BaseDatasetTorch
from core.train.eval import evaluate
from data.data_pipeline.utils import make_torch_collate_fn
from metrics.f1_measure import F1Measure
from metrics.accuracy_metric import AccuracyMetric
from core.train.utils import custom_load_dict


def eval_pipeline(config, model):
    """
    Evaluation pipeline function
    :param config: train config
    :param model: model to evaluate
    """

    # load model
    model_weights = torch.load(config.get_string('model'))
    model = custom_load_dict(model, model_weights)

    # init GPU device
    device = torch.device('cuda')
    # load model into GPU
    model = model.to(device)

    mfcc_transform = T.MFCC(
        sample_rate=44100,
        n_mfcc=20, melkwargs={'n_fft': 2048, 'n_mels': 128, 'hop_length': 512})

    # init data loader
    dataset = BaseDatasetTorch(config.get_string('eval_data'), mfcc_transform)
    dataloader = DataLoader(
        dataset,
        config.get_int('batch_size'),
        shuffle=True,
        collate_fn=make_torch_collate_fn(device, model.shape[-1])
    )
    print('Model loading finished.')
    print('Start evaluation')
    acc_val = evaluate(model, dataloader, AccuracyMetric(), config['num_classes'])
    f1_val = evaluate(model, dataloader, F1Measure(), config['num_classes'])
    print('Evaluation finished.')
