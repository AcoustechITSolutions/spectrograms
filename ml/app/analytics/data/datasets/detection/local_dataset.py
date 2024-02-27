import os
import warnings
import librosa

from data.datasets.algorithms import algorithms


@algorithms.register('detection')
def detection_records(config):
    print('Start data generator script.')
    recs = []
    dataset_path = config.get_string('cache_data')
    for root, _, files in os.walk(dataset_path):
        for name in files:
            info = f"{os.path.relpath(root, dataset_path)}/{name}"
            file_path = os.path.join(root, name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    track, sr = librosa.load(file_path, sr=44100)
                except FileNotFoundError:
                    print(f'Fail to load: {file_path}')
                    continue
                label = 1. if config.get_string('audio_type') in info else 0.
                recs.append({'label': label, 'data': track, 'source': info})
    return recs
