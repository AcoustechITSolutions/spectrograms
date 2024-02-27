import sys

import numpy as np

from tools.speaker_ID.utils import detect, \
    make_records_for_detector, \
    preprocess_wav, \
    embed_utterance, \
    WINDOW_STEP

speech_len_threshold = 3


def make_embed(model, record, sample_rate):
    """

    :param model: embedder model
    :param record: speech audio array
    :param sample_rate: sample rate
    :return: audio embedding
    """
    processed_record = preprocess_wav(record, sample_rate)
    if len(processed_record) / 16000 < speech_len_threshold:
        print('To few speech or not detected')
        sys.exit()
    return embed_utterance(processed_record, model)


def split_audio(record, sample_rate, model=None, point=None):
    """
    Split audio for speech and cough. Uses model or split second.
    :param record: full audio array
    :param sample_rate: sample rate
    :param model: detector model
    :param point: separation point between speech and cough
    :returns two audio arrays
    """
    if point:
        point = point / WINDOW_STEP
        model = None
    if model:
        _, point = detect(model, make_records_for_detector(record, sample_rate))
    speech_ = record[:int(sample_rate * point * WINDOW_STEP)]
    coughing = record[int(sample_rate * point * WINDOW_STEP):]
    return speech_, coughing


def calculate_similarity(emb1, emb2):
    """
    Cosine similarity between two embeddings.
    :param emb1: first embedding
    :param emb2: second embedding
    :return: cosine similarity
    """
    return np.inner(emb1, emb2)

