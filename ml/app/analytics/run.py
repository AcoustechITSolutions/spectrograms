#!/usr/bin/env
import argparse
from core.pipeline import app_pipeline
from core.train.train_pipeline import train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to run training process')

    parser.add_argument('-c', '--config', type=str,
                        required=True,
                        help='path to config file')

    return parser.parse_args()


if __name__ == '__main__':
    print('Start training process.', 'Info prints via TensorBoard', sep='\n')
    args = parse_args()
    app_pipeline(args.config, train_pipeline)
    print('Finish training process.')
