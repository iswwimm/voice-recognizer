import os
import random
import librosa
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt

MY_VOICE_FOLDER = "data/my_voice/sliced_1" 

files = os.listdir(MY_VOICE_FOLDER)
if not files:
    print("Error: Directory is empty.")
    exit()

random_file = random.choice(files)
file_path = os.path.join(MY_VOICE_FOLDER, random_file)

print(f"Processing file: {random_file}")

array, sample_rate = librosa.load(file_path, sr=None) 

waveform = torch.tensor(array).float().unsqueeze(0)

print(f"Shape: {waveform.shape}, SR: {sample_rate}")

mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    win_length=None,
    hop_length=512,
    n_mels=64
)

spec = mel_transform(waveform)
spec = T.AmplitudeToDB()(spec)

plt.figure(figsize=(10, 4))
plt.imshow(spec[0].numpy(), cmap='inferno', origin='lower', aspect='auto')
plt.title(f"Mel Spectrogram: {random_file}")
plt.ylabel("Frequency (Mel)")
plt.xlabel("Time frames")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()