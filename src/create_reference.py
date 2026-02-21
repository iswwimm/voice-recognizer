import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio.transforms as T

from model import SpeakerEncoder

MODEL_PATH = "voice_auth_model.pth"
MY_VOICE_DIR = "data/my_voices"   
OUTPUT_REF_PATH = "my_voice_embedding.pt"
DEVICE = "cpu"

def get_embedding(model, file_path):
    """Extracts and returns the normalized embedding for a given audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        
        target_len = 64 * 512
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
            
        wav = torch.tensor(audio).float().unsqueeze(0)
        
        spec = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)(wav)
        spec = T.AmplitudeToDB()(spec)
        spec = spec.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = model(spec)
            
        return embedding
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    print("Loading model.")
    model = SpeakerEncoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 

    files = [str(p) for p in Path(MY_VOICE_DIR).rglob("*.wav")]
    print(f"Found {len(files)} files for reference embedding generation.")

    embeddings = []
    for path in files:
        emb = get_embedding(model, path)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        print("Error: No valid embeddings could be generated.")
        return
    embeddings = torch.stack(embeddings)
    mean_embedding = torch.mean(embeddings, dim=0)
    mean_embedding = torch.nn.functional.normalize(mean_embedding, p=2, dim=1)

    torch.save(mean_embedding, OUTPUT_REF_PATH)
    print(f"Reference embedding successfully saved to '{OUTPUT_REF_PATH}'")

if __name__ == "__main__":
    main()