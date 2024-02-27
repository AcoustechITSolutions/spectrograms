import io
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

hop_length = 512
n_fft = 2048

def generate_spectrogram(audio: np.ndarray, samplerate: int) -> io.BytesIO:
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig = plt.figure()
   
    librosa.display.specshow(S_db, y_axis=None, x_axis=None, sr=samplerate,
                            hop_length=hop_length, bins_per_octave=100)
    audio_spectre = io.BytesIO()
   
    fig.gca().set_axis_off()
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
               hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(audio_spectre, format='png', bbox_inches='tight', pad_inches=0.0)
    audio_spectre.seek(0)
 
    fig.clear()
    plt.close(fig)
    plt.clf()
    plt.close('all')
    
    return audio_spectre