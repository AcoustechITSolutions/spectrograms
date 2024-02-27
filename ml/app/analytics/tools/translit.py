#!/usr/bin/env
import numpy as np
import pandas as pd
import argparse
import re
import os
import sys
import unicodedata


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to transliterating file names')

    parser.add_argument('-idt', '--input-dataset-table', type=str,
                        help='table with dataset info')

    parser.add_argument('-idp', '--input-dataset-path', type=str,
                        help='dataset path')

    return parser.parse_args()


def get_rus_name(tab_path):
    """
    get indexes of records with russian name in the table
    :param tab_path: path to the table
    :return indexes in the table
    """

    def get_idxs(rec_name, ind):
        if bool(re.search('[а-яА-Я]', rec_name)):
            return ind

    if 'xlsx' in tab_path:
        data = pd.read_excel(tab_path)
    else:
        data = pd.read_csv(tab_path)
    recs = data.iloc[:, 0].to_numpy()
    # amount of lines for table header
    extra_ind = 0
    while type(recs[extra_ind]) == float:
        extra_ind += 1
    rec_names = recs[extra_ind:]
    indxs = np.arange(0, len(rec_names))
    rus_indxs = np.array(list(map(get_idxs, rec_names, indxs)))
    return rus_indxs[rus_indxs != np.array(None)] + extra_ind


def translit_table(tab_path, rus_idxs, csv_fl=True):
    """
    changes russian names of records in the table
    :param tab_path: path to the table
    :param rus_idxs: indexes of necessary records in the table
    :param csv_fl: flag indicating to save the file in .csv format
    :return old full names of records (e.g. /audio/sick/covid/Имя.wav)
    """

    def transliteration(idx):
        cyrillic = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        latin = 'a|b|v|g|d|e|e|zh|z|i|i|k|l|m|n|o|p|r|s|t|u|f|kh|tc|ch|sh|shch||y||e|iu|ia|' \
                'A|B|V|G|D|E|E|Zh|Z|I|I|K|L|M|N|O|P|R|S|T|U|F|Kh|Tc|Ch|Sh|Shch||Y||E|Iu|Ia'.split('|')
        old_nam = data.iloc[idx, 0].partition('.')[2]
        new_nam = unicodedata.normalize('NFC', data.iloc[idx, 0]).translate({ord(k): v for k, v in zip(cyrillic, latin)})
        data.iloc[idx, 0] = new_nam
        return old_nam

    if 'xlsx' in tab_path:
        data = pd.read_excel(tab_path)
    else:
        data = pd.read_csv(tab_path)
    old_rec_names = list(map(transliteration, rus_idxs))
    # save the table
    if csv_fl:
        if 'xlsx' in tab_path:
            data.to_csv(tab_path.rpartition('.')[0] + '.csv', index=False)
        else:
            data.to_csv(tab_path, index=False)
    else:
        data.to_excel(tab_path, index=False)
    return old_rec_names


def translit_files(tab_path, data_path, rus_idxs, old_names):
    """
    changes the russian names of the files themselves
    :param tab_path: path to the table
    :param data_path: path to the directory with dataset
    :param rus_idxs: indexes of necessary records in the table
    :param old_names: old (current) names of files
    """

    def rename_files(list_i, tab_i):
        try:
            os.rename(data_path + old_names[list_i], data_path + data.iloc[tab_i, 0].partition('.')[2])
        except OSError:
            print('File ', data_path + unicodedata.normalize('NFC', old_names[list_i]), 'does not exist')

    data = pd.read_excel(tab_path)
    list_indxs = np.arange(0, len(old_names))
    _ = list(map(rename_files, list_indxs, rus_idxs))


if __name__ == "__main__":
    args = parse_args()
    table_path = args.input_dataset_table
    dataset_path = args.input_dataset_path
    rus_indexes = get_rus_name(table_path)
    if len(rus_indexes) > 0:
        rec_old_names = translit_table(table_path, rus_indexes, csv_fl=False)
        translit_files(table_path, dataset_path, rus_indexes, rec_old_names)
    print('The program is over')
