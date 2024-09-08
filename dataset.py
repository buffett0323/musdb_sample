""" Buffett 0908 """

import musdb
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm


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



# Create dataset and data loader
if __name__ == "__main__":
    mus = musdb.DB(subsets='train', split='train', download=True)
    dataset = MUSDBDataset(mus)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    for i in tqdm(data_loader):
        pass
