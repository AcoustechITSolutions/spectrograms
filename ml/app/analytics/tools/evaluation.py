#!/usr/bin/env
import argparse
from core.pipeline import app_pipeline
from core.train.eval_pipeline import eval_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to convert data into .nyp format')

    parser.add_argument('-m', '--model', type=str,
                        required=True, help='path to model')

    parser.add_argument('-c', '--config', type=str,
                        required=True, help='path to config file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app_pipeline(args.config, eval_pipeline, args.model)
