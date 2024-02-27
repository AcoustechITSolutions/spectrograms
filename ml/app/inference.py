import torch.nn as nn
import librosa
import torch
import dataclasses as dc
import sys, os, math
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as f
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), 'app/analytics/'))

from app.analytics.data.data_pipeline.dataset import BaseDataset
from app.my_dataset import MyDataset
from app.analytics.data.data_pipeline.utils import make_collate_fn
from tools.noiser.noiser import DatasetWithoutLoad

@dc.dataclass()
class InferenceResult:
    prediction: float
    audio_duration: float
    samplerate: int

def pad_spectrogram(data: torch.Tensor, seq_elem_size: int, width: int):
    return f.pad(
        input=data,
        pad=[0, seq_elem_size - width, 0, 0],
        mode='constant',
        value=0
    )

def windowed_inference(model: nn.Module, device: torch.device, file_path: str) -> InferenceResult:
    track, sr = librosa.load(file_path, sr=44100)
    audio_duration = librosa.get_duration(y=track, sr=sr)
    track = librosa.feature.melspectrogram(track, sr)
    seq_numbers = math.ceil(audio_duration / 4)
    if seq_numbers == 0:
        seq_numbers = 1
    seq_elem_size = int(track.shape[1] / seq_numbers)
    
    proc_batch = []
    data = torch.from_numpy(track)
    dim = data[0].shape
    seq_len = max(dim[-1] // seq_elem_size, 1)
    if len(dim) == 2:
        _, w = dim
        # pad too small spectrogram
        if w < seq_elem_size:
            data = pad_spectrogram(data, seq_elem_size, w)
        data = data[:, :, 0:seq_len * seq_elem_size]
    data = list(torch.split(data, seq_elem_size, dim=-1))
    for i in range(0, len(data)):
        item_data = data[i]
        dim = item_data.shape
        _, w = dim
        if w < seq_elem_size:
            item_data = pad_spectrogram(item_data, seq_elem_size, w)
        data[i] = item_data
    data = torch.stack(data)
    proc_batch.append(torch.squeeze(data, dim=1))
    predictions = []
    for item in data:
        shape = item.shape
        # rec = Record(1, sr, shape, item.flatten())
        # data = Data()
        # data.add(rec)
        data = {"label": 1, "data": item}
        dataset = MyDataset(data)
    
        dataloader = DataLoader(
            dataset,
            batch_size = 1,
            collate_fn = make_collate_fn(device, model)
        )
       
        model.eval()
        pred = None
        for idx, batch in enumerate(dataloader):
            if len(batch) == 4:
                _, batch_data, _, pad_lengths = batch
                pred = model.forward((batch_data, pad_lengths))
            else:
                _, batch_data = batch
                pred = model.forward(batch_data)
        print(f'got prediction {pred[0][0].item()}')
        predictions.append(pred[0][0].item())
    mean = np.mean(predictions)
    
    return InferenceResult(prediction = mean, audio_duration = audio_duration,
                           samplerate = sr)

def inference(model: nn.Module, device: torch.device, file_path: str) -> InferenceResult:
    track, sr = librosa.load(file_path, sr=44100)
    audio_duration = librosa.get_duration(y=track, sr=sr)
    data = [{
        'label': 1,
        'data': [librosa.feature.mfcc(track)],
        'source': file_path,
    }]
    dataset = DatasetWithoutLoad(data=data)

    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        collate_fn = make_collate_fn(device, model)
    )
   
    model.eval()
    pred = None
    for idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            _, data, pad_lengths = batch
            pred = model.forward((data, pad_lengths))
        else:
            _, data = batch
            pred = model.forward(data)

    return InferenceResult(prediction = pred.item(), audio_duration = audio_duration,
                           samplerate = sr)
