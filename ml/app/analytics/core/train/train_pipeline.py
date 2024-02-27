#!/usr/bin/env
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch_audiomentations import Compose, Gain, Shift

from torch.utils.data import DataLoader
from core.train.train import train
from core.train.optimizer import create_optimizer
from data.data_pipeline.dataset import BaseDatasetTorch
from metrics.accuracy_metric import AccuracyMetric
from data.data_pipeline.utils import make_torch_collate_fn
from core.train.utils import custom_load_dict, WeightedBCE


def train_pipeline(config, model):
    """
    This is function is responsible for creating all necessary entities
    for training and calling train process itself
    :param config: training configuration
    :param model: pytorch model
    """
    # define metric
    metric = AccuracyMetric()

    mfcc_transform = T.MFCC(
        sample_rate=44100,
        n_mfcc=20, melkwargs={'n_fft': 2048, 'n_mels': 128, 'hop_length': 512})

    apply_augmentation = Compose(
        transforms=[
            Gain(min_gain_in_db=-5.0, max_gain_in_db=5.0, p=0.3),
            Shift(min_shift=-2, max_shift=2, p=0.5)
        ])

    # init dataset handlers
    train_dataset = BaseDatasetTorch(config['train_data'], mfcc_transform, apply_augmentation)
    eval_dataset = BaseDatasetTorch(config['eval_data'], mfcc_transform)
    print(f'Size of training dataset: {len(train_dataset)}')
    pos_count, neg_count = train_dataset.pos_items(), train_dataset.neg_items()
    print(f'Positive samples: {pos_count}   Negative samples: {neg_count}')
    print(f'Size of evaluation dataset: {len(eval_dataset)}')

    # define loss
    if config.get_int('num_classes') == 1:
        criterion = WeightedBCE([1, round(neg_count/pos_count, 1)])
    else:
        criterion = nn.CrossEntropyLoss()

    # init GPU device
    device = torch.device('cuda')

    if 'checkpoint' in config:
        model_weights = torch.load(config['checkpoint'])
        model = custom_load_dict(model, model_weights)

    # load model into GPU
    model = model.to(device)
    # create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        config.get_int('batch_size'),
        shuffle=False,
        collate_fn=make_torch_collate_fn(device, model.shape[-1])
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        config.get_int('batch_size'),
        shuffle=True,
        collate_fn=make_torch_collate_fn(device, model.shape[-1])
    )

    num_epochs = config['num_epochs']
    print(f'Total epoch: {num_epochs}')

    # make optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)

    # run training process
    train(model, train_dataloader, eval_dataloader,
          optimizer, scheduler, criterion, metric,
          config.get_string('model_name'), num_epochs,
          config.get_int('eval_step_epochs'), config.get_int('epochs_to_save_model'),
          config.get_int('num_classes'))
