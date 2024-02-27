import librosa
import numpy as np
import webrtcvad
import struct
from scipy.ndimage.morphology import binary_dilation

from tools.speaker_ID.cons import *


WINDOW_STEP = 0.5
SIGNAL_DURATION = 1
freq_THRESHOLD = 0.6
int16_max = (2 ** 15) - 1


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
    for i in range(len(record)):
        ort_outs = model.run(None,
                             {'input_1': np.expand_dims(record[i], axis=(1, 0))}
                             )
        if round(ort_outs[0].item()) == 1:
            if count == 0:
                tag = i
            count += 1
    return count, tag


def preprocess_wav(wav, source_sr):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that
    The waveform will be resampled to match the data hyperparameters.

    :param wav: audio array
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    """

    # Resample the wav
    if source_sr != 16000:
        wav = librosa.resample(wav, source_sr, 16000)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)

    return wav


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def compute_partial_slices(n_samples: int, rate, min_coverage):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to
    obtain partial utterances of <partials_n_frames> each. Both the waveform and the
    mel spectrogram slices are returned, so as to make each partial utterance waveform
    correspond to its spectrogram.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

    :param n_samples: the number of samples in the waveform
    :param rate: how many partial utterances should occur per second. Partial utterances must
    cover the span of the entire utterance, thus the rate should not be lower than the inverse
    of the duration of a partial utterance. By default, partial utterances are 1.6s long and
    the minimum rate is thus 0.625.
    :param min_coverage: when reaching the last partial utterance, it may or may not have
    enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
    then the last partial utterance will be considered by zero-padding the audio. Otherwise,
    it will be discarded. If there aren't enough frames for one partial utterance,
    this parameter is ignored so that the function always returns at least one slice.
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
    respectively the waveform and the mel spectrogram with these slices to obtain the partial
    utterances.
    """
    assert 0 < min_coverage <= 1

    # Compute how many frames separate two partial utterances
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
    assert 0 < frame_step, "The rate is too high"
    assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
                                            (sampling_rate / (samples_per_frame * partials_n_frames))

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partials_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partials_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_utterance(wav: np.ndarray, ort_session, return_partials=False, rate=1.3, min_coverage=0.75):

    # Compute where to split the utterance into partials and pad the waveform with zeros if
    # the partial utterances cover a larger range.
    wav_slices, mel_slices = compute_partial_slices(len(wav), rate, min_coverage)
    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Split the utterance into partials and forward them through the model
    mel = wav_to_mel_spectrogram(wav)
    mels = np.array([mel[s] for s in mel_slices])

    ort_outs = ort_session.run(None, {
        'input_1': mels
    })
    partial_embeds = ort_outs[0]

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wav_slices
    return embed