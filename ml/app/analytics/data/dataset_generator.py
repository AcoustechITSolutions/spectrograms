#!/usr/bin/env
from functools import partial
from pathlib import Path
from random import shuffle
import librosa
import numpy as np

from data.augmentation import augment_train_data
from data.utils import remove_file_if_existed
from data.datasets.algorithms import algorithms


def generate_dataset_binary(config):

    dataset_name = algorithms.get(config.get_string('name'))

    records = dataset_name(config)

    shuffle(records)
    pos = [rec for rec in records if rec['label'] == 1]
    neg = [rec for rec in records if rec['label'] == 0]
    print(f'Positive samples: {len(pos)}\n'
          f'Negative samples: {len(neg)}\n' 
          f'Before Augmentation')
    eval_records = pos[:int(config.get_int('eval_size') / 2)] + neg[:int(config.get_int('eval_size') / 2)]
    train_records = pos[int(config.get_int('eval_size') / 2):] + neg[int(config.get_int('eval_size') / 2):]
    shuffle(eval_records)
    shuffle(train_records)
    # augment train data if needed
    if config.get_bool('balance_classes'):
        print('Start data augmentation.')
        if 'spectrogram' not in config:
            raise NotImplementedError
        train_records = augment_train_data(
            train_records,
            config.get_int('augmentation_size')
            )

    spectrogram_fns = {'stft': partial(librosa.stft, n_fft=512),
                       'mfcc': librosa.feature.mfcc,
                       'mel': librosa.feature.melspectrogram}

    shuffle(train_records)

    for idx in range(len(train_records)):
        for n in range(len(train_records[idx]['data'])):
            train_records[idx]['data'][n] = spectrogram_fns[config.get_string('spectrogram')](train_records[idx]['data'][n])
    for idx in range(len(eval_records)):
        for n in range(len(eval_records[idx]['data'])):
            eval_records[idx]['data'][n] = spectrogram_fns[config.get_string('spectrogram')](eval_records[idx]['data'][n])

    train_output = Path(config.output) / 'train.npy'
    eval_output = Path(config.output) / 'eval.npy'
    print('Train binary file generation start.')
    remove_file_if_existed(train_output)
    np.save(Path(config.output) / 'train.npy', train_records)
    print('Eval binary file generation start')
    remove_file_if_existed(eval_output)
    np.save(eval_output, eval_records)

    print(f'Dataset generation finished. It\'s saved here: {config.output}')
    return str(train_output), str(eval_output)


def generate_dataset_multi(config):

    dataset_name = algorithms.get(config.get_string('name'))

    records = dataset_name(config)

    shuffle(records)
    if config['cough_char'] == 'intensity':
        cl1 = [rec for rec in records if rec['label'] == 0]
        cl2 = [rec for rec in records if rec['label'] == 1]
        cl3 = [rec for rec in records if rec['label'] == 2]

    print(f'Paroxysmal samples: {len(cl1)} \n'
          f'Paroxysmal_hacking samples: {len(cl2)} \n'
          f'Not paroxysmal samples: {len(cl3)}')

    eval_records = cl1[:int(len(cl1) / 7)] + cl2[:int(len(cl2) / 7)] + cl3[:int(len(cl3) / 7)]
    train_records = cl1[int(len(cl1) / 7):] + cl2[int(len(cl2) / 7):] + cl3[int(len(cl3) / 7):]
    shuffle(eval_records)
    shuffle(train_records)
    # augment train data if needed
    if config.get_bool('balance_classes'):
        print('Start data augmentation.')
        if 'spectrogram' not in config:
            raise NotImplementedError
        train_records = augment_train_data(
            train_records,
            config.get_int('augmentation_size')
            )

    spectrogram_fns = {'stft': partial(librosa.stft, n_fft=512),
                       'mfcc': librosa.feature.mfcc,
                       'mel': librosa.feature.melspectrogram}

    shuffle(train_records)

    for idx in range(len(train_records)):
        for n in range(len(train_records[idx]['data'])):
            train_records[idx]['data'][n] = spectrogram_fns[config.get_string('spectrogram')](train_records[idx]['data'][n])
    for idx in range(len(eval_records)):
        for n in range(len(eval_records[idx]['data'])):
            eval_records[idx]['data'][n] = spectrogram_fns[config.get_string('spectrogram')](eval_records[idx]['data'][n])

    train_output = Path(config.output) / 'train.npy'
    eval_output = Path(config.output) / 'eval.npy'
    print('Train binary file generation start.')
    remove_file_if_existed(train_output)
    np.save(Path(config.output) / 'train.npy', train_records)
    print('Eval binary file generation start')
    remove_file_if_existed(eval_output)
    np.save(eval_output, eval_records)

    print(f'Dataset generation finished. It\'s saved here: {config.output}')
    return str(train_output), str(eval_output)
