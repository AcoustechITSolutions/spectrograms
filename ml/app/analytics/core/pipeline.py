#!/usr/bin/env
from typing import Callable
from pyhocon import ConfigFactory
from data.dataset_generator import generate_dataset_binary, generate_dataset_multi
from models.algorithms import algorithms


def app_pipeline(path_to_config, pipeline_fn: Callable, path_to_ckpt=None):
    """
    Execute application pipeline according to config file
    :param path_to_config: path to configuration file
    :param pipeline_fn: function to execute in pipeline
    :param path_to_ckpt: path to predefined model (optional)
    """
    data_conf, model_conf, preproc_conf, train_conf = prepare_config(path_to_config)

    if 'generated' in data_conf:
        train_conf.put('train_data', data_conf.get_string('generated.train_data'))
        train_conf.put('eval_data', data_conf.get_string('generated.eval_data'))
    else:
        if 'cough_char' not in data_conf:
            train_output, eval_output = generate_dataset_binary(data_conf)
        else:
            train_output, eval_output = generate_dataset_multi(data_conf)
        train_conf.put('train_data', train_output)
        train_conf.put('eval_data', eval_output)

    train_conf.put('input_shape', model_conf.get_list('args.input_shape'))
    train_conf.put('model_name', model_conf.get_string('name'))
    train_conf.put('num_classes', model_conf.get_int('args.num_classes'))
    if path_to_ckpt:
        train_conf.put('model', path_to_ckpt)
    model = algorithms.get(model_conf.get_string('name'))(**model_conf['args'])

    pipeline_fn(train_conf, model)


def prepare_config(path):
    """
    Read config file
    :param path: path to config file
    :return tuple with configurations for necessary modules
    """
    conf = ConfigFactory.parse_file(path)
    preproc_conf = conf.get_config('preprocessing', None)
    return conf['data'], conf['model'], preproc_conf, conf['train']
