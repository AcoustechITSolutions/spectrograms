#!/usr/bin/env
import numpy as np
from data.utils import compose_aug


def augment_train_data(records, samples_per_class=1000):
    """
    Balance number of samples per class and adds new samples by augmentation
    :param records: list of Record class instance with train data
    :param samples_per_class: number of samples each class should have
     (this number assumed bigger than any class size)
    :return bigger list of records
    """

    labels = {}
    for idx, rec in enumerate(records):
        if rec['label'] in labels:
            labels[rec['label']].append(idx)
        else:
            labels[rec['label']] = [idx]

    new_records = []
    for label, samples in labels.items():
        # do not generate extra samples for already well sampled class
        if samples_per_class <= len(samples):
            continue
        samples_to_make = samples_per_class - len(samples)
        for decision in np.random.randint(2, size=samples_to_make):
            idx = np.random.randint(len(samples), size=1)[0]
            sample = records[samples[idx]]

            data = []
            for i in range(len(sample['data'])):
                data.append(compose_aug(sample['data'][i]))

            new_records.append({'label': label,
                                'data': data,
                                'source': sample['source']
                                })

    return records + new_records
