import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

class VoiceTripletDataset(Dataset):
    def __init__(self, my_voice_dir, other_voices_dir, sample_rate=16000, max_len=64):
        """
        Args:
            my_voice_dir (str): Root directory containing the target speaker's audio (searched recursively).
            other_voices_dir (str): Root directory containing other speakers' audio.
            sample_rate (int): Target sample rate for audio processing.
            max_len (int): Maximum width of the spectrogram in time frames.
        """
        self.my_files = [str(p) for p in Path(my_voice_dir).rglob("*.wav")]
        self.other_files = [str(p) for p in Path(other_voices_dir).rglob("*.wav")]
        
        print("Dataset Statistics:")
        print(f"  Target voice: {len(self.my_files)} files")
        print(f"  Other voices: {len(self.other_files)} files")

        self.sample_rate = sample_rate
        self.max_len = max_len
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.my_files)

    def preprocess(self, file_path):
        """Loads audio and converts it to a Mel-spectrogram tensor."""
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate)           
            target_samples = self.max_len * 512 
            
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')

            waveform = torch.tensor(audio).float().unsqueeze(0) 
            
            spec = self.mel_transform(waveform)
            spec = self.db_transform(spec)
            
            return spec 
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return torch.zeros(1, 64, self.max_len)

    def __getitem__(self, idx):
        anchor_path = self.my_files[idx]
        
        positive_path = random.choice(self.my_files)
        for _ in range(10):
            if positive_path != anchor_path: 
                break
            positive_path = random.choice(self.my_files)
            
        negative_path = random.choice(self.other_files)
        
        return (
            self.preprocess(anchor_path),
            self.preprocess(positive_path),
            self.preprocess(negative_path)
        )

if __name__ == "__main__":
    dataset = VoiceTripletDataset(
        my_voice_dir="data/my_voices", 
        other_voices_dir="data/other_voices"
    )
    
    if len(dataset) > 0:
        anc, pos, neg = dataset[0]
        print(f"Tensors ready. Anchor shape: {anc.shape}")
    else:
        print("Dataset is empty. Please verify the provided directory paths.")