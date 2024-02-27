import torch

import librosa
import librosa.feature

def wav_questible(preds, data, idx, path_to_save, spec):
    data_np = data.cpu().numpy()
    print(f'Start generate questible tracks for batch {idx + 1}')
    for bt in range(len(preds)):
        if (preds[bt].cpu().numpy()[0] < 0.6) and (preds[bt].cpu().numpy()[0] > 0.4):
            for idxx in range(data_np.shape[0]):
                if idxx == 0:
                    rec = data[idxx][bt]
                else:
                    rec = torch.cat((rec, data[idxx][bt]), 1)
            if spec == 'mfcc':
                track = librosa.feature.inverse.mfcc_to_audio(rec.cpu().numpy())
                librosa.output.write_wav(path_to_save + f"questible_{idx + 1}_{bt}_{round(float(preds[bt].cpu().numpy()[0]), 3)}.wav", track, sr=44100)

            elif spec == 'mel':
                track = librosa.feature.inverse.mel_to_audio(rec.cpu().numpy())
                librosa.output.write_wav(path_to_save + f"questible_{idx + 1}_{bt}_{round(float(preds[bt].cpu().numpy()[0]), 3)}.wav", track, sr=44100)
            else:
                raise ValueError("Wrong spectrogram type. Can't save samples")

    print('Saved questible')

def wav_mistaken(preds, data, idx, label, path_to_save, spec):
    print(f'Start generate mistaken tracks for batch {idx + 1}')
    data_np = data.cpu().numpy()
    for bt in range(len(preds)):
        if int(preds[bt]) != int(label[bt]):
            for idxx in range(data_np.shape[0]):
                if idxx == 0:
                    rec = data[idxx][bt]
                else:
                    rec = torch.cat((rec, data[idxx][bt]), 1)
            if spec == 'mfcc':
                track = librosa.feature.inverse.mfcc_to_audio(rec.cpu().numpy())
                librosa.output.write_wav(path_to_save + f"mistaken_{idx + 1}_{bt}.wav", track, sr=44100)
            elif spec == 'mel':
                track = librosa.feature.inverse.mel_to_audio(rec.cpu().numpy())
                librosa.output.write_wav(path_to_save + f"mistaken_{idx + 1}_{bt}.wav", track, sr=44100)
            else:
                raise ValueError("Wrong spectrogram type. Can't save samples")
    print('Saved mistaken')