#!/usr/bin/env
import os
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift
from tabulate import tabulate as tb
from sqlalchemy.orm import sessionmaker
from botocore.exceptions import ClientError

from data.db_utils import get_engine_for_port,\
                          create_bucket,\
                          ssh_forwarder,\
                          get_s3_key,\
                          BUCKET_NAME


def add_white_noise(mel_spec, noise_factor=.005):
    """
    Applies white noise to spectrogram for data augmentation
    :param mel_spec: np array with mel spectrogram
    :param noise_factor: scale factor for noise amplitude
    :return spectrogram with noised image
    """
    noise = np.random.random(mel_spec.shape)
    return (mel_spec + noise_factor * noise).astype(np.float32)


def spectrogram_emda(mel_spec1, mel_spec2, alpha=0.5):
    """
    Compute averaging sum between two spectrogram.
    If one spectrogram is shorter than another the part of the
    second one will be used for this time domain.
    Frequency domain should be the same for both spectrogram.
    :param mel_spec1: np array with mel spectrogram
    :param mel_spec2: np array with mel spectrogram
    :param alpha: float between (0, 1) with convex sum factor
    :return mixed spectrogram
    """
    w1, w2 = mel_spec1.shape[1], mel_spec2.shape[1]
    w = min(w1, w2)

    res = np.zeros((mel_spec1.shape[0], max(w1, w2)), dtype=np.float32)
    res[:, 0:w] = alpha * mel_spec1[:, 0:w] + (1. - alpha) * mel_spec2[:, 0:w]

    if w1 < w2:
        res[:, w1:] = mel_spec2[:, w1:]
    elif w1 > w2:
        res[:, w2:] = mel_spec1[:, w2:]

    return res.astype(np.float32)


def remove_file_if_existed(path):
    """
    Helper function to remove existing files or do nothing if it doesn't exist
    :param path: path to file
    """
    if not os.path.isfile(path):
        return
    try:
        os.remove(path)
        print(f'File {path} was deleted')
    except OSError:
        pass


def compose_aug(arr):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    augmented_sample = augment(samples=arr, sample_rate=44100)

    return augmented_sample


def metric_request(keys):
    return f"""
        select
            audio.audio_path,
            m_status.marking_status,
            d_status.marking_status,
            audio.is_representative,
            users.login as data_source,
            cough_char.commentary,
            covid19.symptomatic_type,
            audio.samplerate
        from dataset_audio_info as audio
        join dataset_cough_characteristics as cough_char
            on audio.request_id = cough_char.request_id
        join dataset_request as req
            on audio.request_id = req.id
        join dataset_patient_diseases as patient
            on req.id = patient.request_id
        join covid19_symptomatic_types as covid19
            on covid19.id = patient.covid19_symptomatic_type_id
        join dataset_audio_types as types
            on types.id = audio.audio_type_id
        join dataset_marking_status as m_status
            on m_status.id = req.marking_status_id
        join dataset_marking_status as d_status
            on d_status.id = req.doctor_status_id
        join users
            on users.id = req.user_id
        where audio.audio_path in ({", ".join(keys)})
    """


def make_meta_table(db, e_data, meta_confusion_types, download):
    meta_table = []
    headers = ['audio path', 'metric', 'marking\nstatus', 'doctor\nstatus', 'represent',
               'data\nsource', 'commentary', 'symptomatic\ntype', 'samplerate']

    if download is not None:
        print('Load data from Amazon S3 storage.')
        for confusion_types in meta_confusion_types:
            if not os.path.isdir(f'{download}/{confusion_types}'):
                os.makedirs(f'{download}/{confusion_types}')

    for idx, data in enumerate(e_data):
        if data[1] in meta_confusion_types:
            meta = list(db[idx])
            meta.insert(1, data[1])
            meta_table.append(meta)
            if download is not None:
                try:
                    filename = os.path.basename(meta[0])
                    filename = f'{download}/{data[1]}/{idx}_{meta[5]}_{filename}'
                    download_audio_from_metric(filename, meta[0])
                except ClientError:
                    print(f'Audio file {meta[0]} was not found in AWS bucket.')
                    os.remove(filename)

    return tb(meta_table, headers=headers, tablefmt="fancy_grid")


def generate_classification_metric(config, keys, uncertainties, confusion_types=None, download=None):
    with ssh_forwarder(config) as tunnel:
        tunnel.start()
        engine = get_engine_for_port(tunnel.local_bind_port)
        session = sessionmaker(bind=engine)()
        db_info = session.execute(metric_request(keys)).fetchall()
        session.close()

    source = [f"'s3://acoustery/{s3}'" for s3 in keys]
    eval_data = [list(tup) for tup in zip(source, uncertainties)]
    eval_data.sort()
    db_info.sort()

    if confusion_types is not None:
        return make_meta_table(db_info, eval_data, confusion_types, download)
    return make_source_table(db_info, eval_data)


def make_source_table(db, e_data):
    cm_table = [['tp', 'tn', 'fp', 'fn']]
    headers = ['confusion matrix']
    confusion_matrix = {'tp': 0, 'tn': 1, 'fp': 2, 'fn': 3}

    for idx, data in enumerate(e_data):
        meta = list(db[idx])
        if meta[4] not in headers:
            headers.append(meta[4])
            cm_table.append([0] * 4)
        cm_table[headers.index(meta[4])][confusion_matrix[data[1]]] += 1
    headers.append('total')
    cm_table.append(np.array(cm_table[1:]).sum(axis=0).tolist())
    return tb(np.array(cm_table).T.tolist(), headers=headers, tablefmt="fancy_grid")


def generate_detection_metric(keys, uncertainties, meta_confusion_types):
    cm_table = [['tp', 'tn', 'fp', 'fn']]
    headers = ['confusion matrix']
    confusion_matrix = {'tp': 0, 'tn': 1, 'fp': 2, 'fn': 3}

    e_data = [list(tup) for tup in zip(keys, uncertainties)]

    for path, key in e_data:
        if key in meta_confusion_types:
            parent_dir = lambda path: os.path.abspath(os.path.join(path, os.pardir))
            if parent_dir(path) not in headers:
                headers.append(parent_dir(path))
                cm_table.append([0] * 4)
            cm_table[headers.index(parent_dir(path))][confusion_matrix[key]] += 1
            print(path)
    headers.append('total')
    cm_table.append(np.array(cm_table[1:]).sum(axis=0).tolist())

    return tb(np.array(cm_table).T.tolist(), headers=headers, tablefmt="fancy_grid")


def download_audio_from_metric(filename, path):
    s3_key = get_s3_key(path, BUCKET_NAME)
    bucket = create_bucket()

    with open(filename, 'wb') as f:
        bucket.download_fileobj(s3_key, f)
