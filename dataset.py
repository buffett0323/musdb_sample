""" Buffett 0908 """

import musdb
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm


import random
from pathlib import Path
from typing import List

import musdb
import torch
from torch.utils.data import Dataset


SET_LENGTH = 261888


# Function to convert audio to spectrogram
def audio_to_spectrogram(audio, n_fft=1022, hop_length=512):
    audio = audio[:SET_LENGTH]
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spec)
    return magnitude



# Function to convert spectrogram back to audio
def spectrogram_to_audio(spec, hop_length=512):
    return librosa.istft(spec, hop_length=hop_length)



# DataLoader to prepare batches of audio data
class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(self, musdb):
        self.musdb = musdb
        
    def __len__(self):
        return len(self.musdb)

    def __getitem__(self, idx):
        track = self.musdb[idx]
        audio_mix = track.audio.sum(axis=1)  # Stereo to mono mix
        audio_vocals = track.targets['vocals'].audio.sum(axis=1)  # Stereo to mono vocals
        
        # Convert audio to spectrogram
        spec_mix = audio_to_spectrogram(audio_mix)
        spec_vocals = audio_to_spectrogram(audio_vocals)
        
        # Convert to PyTorch tensors
        mix_tensor = torch.tensor(spec_mix).unsqueeze(0).float()  # Add channel dimension
        vocals_tensor = torch.tensor(spec_vocals).unsqueeze(0).float()
        
        return mix_tensor, vocals_tensor



class MusdbDataset(Dataset):
    def __init__(
        self, root="data/musdb18-wav", is_train: bool = True, targets: List[str] = None
    ) -> None:
        super().__init__()
        root = Path(root)
        assert root.exists(), f"Path does not exist: {root}"
        self.mus = musdb.DB(
            root=root,
            subsets=["train" if is_train else "test"],
            is_wav=True,
        )
        self.targets = [s for s in targets] if targets else ["vocals", "accompaniment"]

    def __len__(self) -> int:
        return len(self.mus)

    def __getitem__(self, index):
        track = self.mus.tracks[index]
        track.chunk_duration = 5.0
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        x_wav = torch.torch.tensor(track.audio.T, dtype=torch.float32)
        y_target_wavs = {
            name: torch.tensor(track.targets[name].audio.T, dtype=torch.float32)
            for name in self.targets
        }
        # original audio (x) and stems (y == targets)
        return x_wav, y_target_wavs
    
    

# Create dataset and data loader
if __name__ == "__main__":
    mus = musdb.DB(subsets='train', split='train', download=True)
    dataset = MUSDBDataset(mus)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    for i in tqdm(data_loader):
        pass
