import time
import argparse
import sys

import numpy as np
import librosa
import onnxruntime
import noisereduce as nr
from torchaudio.transforms import MFCC
from torch import tensor

from tools.full_api.utils import preprocess_wav, embed_utterance


st = time.time()
WINDOW_STEP = 0.5
SIGNAL_DURATION = 1
freq_THRESHOLD = 0.6
noise_THRESHOLD = 0.8
lower_noise_THRESHOLD = 0.7


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to run training process')

    parser.add_argument('-a', '--audio', type=str,
                        required=True,
                        help='path to audio file')

    parser.add_argument('-c', '--classifier', type=str,
                        required=True,
                        help='path to main classifier model')

    parser.add_argument('-n', '--noiser', type=str,
                        required=True,
                        help='path to noiser model')

    parser.add_argument('-d', '--detector', type=str,
                        required=True,
                        help='path to detector model')

    return parser.parse_args()


def make_records_for_detector(rec, sample_rate):
    """
    Generates list of records for detector
    :param rec:(array) librosa audio waveform
    :param sample_rate:(int) sample rate
    :return:(list) list of short audio samples
    """
    sample_window_step = WINDOW_STEP * sample_rate
    sample_signal_dur = SIGNAL_DURATION * sample_rate

    start_samples = [int(sample) for sample in np.arange(0, len(rec), sample_window_step)
                     if sample + sample_signal_dur < len(rec)]

    tracks = []
    for start in start_samples:
        tracks.append(librosa.feature.mfcc(rec[start:start + sample_signal_dur]))

    return tracks


def detect(model, record):
    """
    Detector main function. Try to detect cough in all records
    :param model:(ONNX) detector onnx model
    :param record:(ndarray) audio mfcc spectrogram
    :return:(bool) cough detector prediction
    """
    count = 0
    for item in record:
        ort_outs = model.run(None,
                             {'input_1': np.expand_dims(item, axis=(1, 0))}
                             )
        if round(ort_outs[0].item()) == 1:
            count += 1
    return count


def make_input(in_track):
    """
    Generate input batch
    :param in_track:(array) audio waveform
    :return:(ndarray) input batch
    """
    mfcc_transform = MFCC(
        sample_rate=44100,
        n_mfcc=20, melkwargs={'n_fft': 2048, 'n_mels': 128, 'hop_length': 512})

    spec = mfcc_transform(tensor(in_track).unsqueeze(0))
    spec = spec[0].numpy()
    out = []
    w = 256
    new_spec = np.pad(spec,
                      ((0, 0), (0, w - (spec.shape[1] - w * int(spec.shape[1] / w)))),
                      'constant',
                      constant_values=0).astype('float32')
    for i in range(round(new_spec.shape[1] / w)):
        out.append(np.expand_dims(new_spec[:, i * w:(i + 1) * w], axis=(1, 0)))

    return np.concatenate(out, axis=0)


def noise_classification(model, default_track, denoised_track, sr):
    """
    Noise classification main function. Predicts noise in audio
    :param model:(ONNX) embedder onnx model
    :param default_track:(ndarray) input waveform
    :param denoised_track:(ndarray) denoised waveform
    :param sr:(int) sample rate
    :return:(float) noise classification prediction
    """

    default_embed = embed_utterance(model, preprocess_wav(default_track, sr))
    denoised_embed = embed_utterance(model, preprocess_wav(denoised_track, sr))

    return np.inner(default_embed, denoised_embed)


def covid_classification(model, batch):
    """
    Covid classifier main function
    :param model:(ONNX) covid classifier onnx model
    :param batch:(ndarray) input batch
    :return:(float) covid classifier prediction
    """
    ort_outs = model.run(None, {'input_1': batch})

    return ort_outs[0].item()


def waveform_noise_reduction(in_track, sr):
    """
    Audio denoising function
    :param in_track:(array) audio waveform
    :param sr:(int) sample rate
    :return:(array) denoised audio waveform
    """
    return nr.reduce_noise(y=in_track,
                           sr=sr,
                           n_std_thresh_stationary=1.3,
                           use_tqdm=True).astype('float32')


if __name__ == '__main__':
    args = parse_args()
    sa = time.time()
    covid_model = onnxruntime.InferenceSession(args.classifier)
    embedder_model = onnxruntime.InferenceSession(args.noiser)
    detector_model = onnxruntime.InferenceSession(args.detector)
    track, sr = librosa.load(args.audio, sr=44100)
    print("Loading audio file and models time: %.2f seconds" % (time.time() - sa))

    if len(track) < 66500:
        print(f'Records is too short. {round(len(track)/sr, 3)} seconds\n'
              f'Try at least 2 seconds\n'
              f'But recommended record length is 10 seconds')
        print("Inference time: %.2f seconds" % (time.time() - st))
    else:
        detect_counts = detect(detector_model,
                               make_records_for_detector(track, sr))
        freq = detect_counts / (len(track) / sr)

        if freq == 0:
            print(f'Cough not detected on record\n'                  
                  f'Please try again')
            sys.exit()
        elif freq < freq_THRESHOLD:
            print(f'There are too few entries for {round(len(track)/sr,1)} seconds record\n'
                  f'Classification quality can get worse')

        track_redu = waveform_noise_reduction(track, sr)

        noise_predict = noise_classification(embedder_model, track, track_redu, sr)

        if noise_predict < lower_noise_THRESHOLD:
            print('Record is too noisy\n'
                  'Please try again')
            sys.exit()

        elif noise_predict < noise_THRESHOLD:
            print('Record may contains noise\n'
                  'Classification quality can get worse')

        covid_predict = covid_classification(covid_model, make_input(track))

        print(f'Covid prediction: {round(covid_predict*100, 1)}%')
        print("Inference time: %.2f seconds" % (time.time() - st))
