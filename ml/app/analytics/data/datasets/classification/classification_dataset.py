import warnings
import librosa

from data.db_utils import load_db_metadata, download_audio_files
from data.datasets.algorithms import algorithms


@algorithms.register('classification')
def classification_records(config):
    print('Start data generator script.\n'
          'Request to database.')
    db_metadata = load_db_metadata(config)
    print('Load data from Amazon S3 storage.')
    metadata = download_audio_files(config, db_metadata)
    print('Loading is finished.\n'
          'Reading audio files.')
    num = int((len(db_metadata[0]) - 2)/3)
    recs = []
    arrays = []
    sources = []
    for idx, item in enumerate(metadata.items()):
        s3_key, data = item
        sources.append(s3_key)
        if idx % 100 == 0 and idx != 0:
            print(f'Read {idx} audio files.')
        # suppress internal librosa lib warning due to .mp3 files opening
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            file_path = data['file_path']

            try:
                track, sr = librosa.load(file_path, sr=44100)

                if data['start'] is not None:
                    time_steps = []
                    for i in range(len(data['start'])):
                        begin = min(int(float(data['start'][i]) * sr), len(track) - 1)
                        end = min(int(float(data['end'][i]) * sr) - 1, len(track) - 1)
                        time_steps += list(range(begin, end))
                    track = track[time_steps]
                arrays.append(track)
            except FileNotFoundError:
                print(f'Fail to load: {file_path}')
                continue
            except ValueError:
                print(f'Corrupted file: {file_path}')
                continue
            if 'cough_char' in config and config['cough_char'] == 'intensity':
                label = int(data['symptomatic']) - 1
            else:
                label = 1. if config.get_bool('force_labels', False)\
                    else data['symptomatic'] not in ['no_covid19', 'none']
            if (1 + idx) % num == 0:
                recs.append({'label': label, 'data': arrays, 'source': sources})
                arrays = []
                sources = []

    return recs