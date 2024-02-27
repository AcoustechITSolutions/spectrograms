#!/usr/bin/env
import argparse
from pathlib import Path
import librosa
import warnings

from data.utils import make_dataframe

"""
    This script return one shallowed subtrack from one track and
    saves it to another directory under the same name.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script return one shallowed subtrack from one track')

    parser.add_argument('-i', '--input-dataset-table', type=str,
                        help='table with dataset info')
    parser.add_argument('-c', '--cut-cough', action='store_true',
                        required=False, default=False,
                        help='cut cough track')
    parser.add_argument('-il', '--interval', type=float,
                        required=False,
                        default=0.2,
                        help='start and end cropping interval')

    return parser.parse_args()


def cut_track_from_row(df_row):
    if args.cut_cough:
        cough_series = [tuple(cough_ts.split('-'))
                        for cough_ts in df_row.cough_series.split()]

        start_of_series = float(cough_series[0][0])
        end_of_series = float(cough_series[len(cough_series) - 1][1])
        audio_type_path = 'cough_audio'
    else:
        inhales_series = [tuple(breath_ts.split('-'))
                          for breath_ts in df_row.inhales_series.split()]
        exhale_series = [tuple(breath_ts.split('-'))
                         for breath_ts in df_row.exhale_series.split()]
        start_of_series = float(inhales_series[0][0])
        end_of_series = float(exhale_series[len(exhale_series) - 1][1])
        audio_type_path = 'breath_audio'

    input_audio_path = Path(args.input_dataset_table).parent / str(df_row.array[0])
    output_audio_path = Path(args.input_dataset_table).parent / audio_type_path / str(df_row.array[0])
    print(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            track, sr = librosa.load(input_audio_path, sr=44100)
        except OSError:
            print('File ', input_audio_path, 'does not exist')
            return
        start = start_of_series - args.interval if start_of_series > args.interval else 0
        end = end_of_series + args.interval
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        cut_track = track[start_sample:] if end_sample > len(track) else track[start_sample:end_sample]
        librosa.output.write_wav(output_audio_path, cut_track, sr=44100)


def cut_tracks(df):
    for idx, (_, row) in enumerate(df.iterrows()):
        cut_track_from_row(row)
        if idx != 0 and idx % 10 == 0:
            print(f'Processed {idx} dataframe rows')


if __name__ == "__main__":
    args = parse_args()
    print('Start data frame initializing.')
    print('Data frame initialized')
    data_frame = make_dataframe(args.input_dataset_table, is_cough=args.cut_cough)
    cut_tracks(data_frame)
    print('The program is over')