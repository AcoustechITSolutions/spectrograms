import pytest
import io
import librosa
from app.audio_processing import generate_spectrogram

# this test is only for visual check
def test_spectrogram():
    with open('./tests/test_audio/cough_audio.wav', 'rb') as f:
        (sig, rate) = librosa.load(f, sr=44100)
        spectrogram = generate_spectrogram(sig, rate)
        with open('./tests/test_spectre.png', 'wb') as s:
            s.write(spectrogram.getbuffer())
