import time
import librosa
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T

from model import SpeakerEncoder

MODEL_PATH = "voice_auth_model.pth"
REF_PATH = "my_voice_embedding.pt"
THRESHOLD = 0.93  
DEVICE = "cpu"
FS = 16000
DURATION = 3

def get_input_device_id():
    """Locate the specific Arctis headset using the MME host API."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "Arctis" in dev['name'] and dev['max_input_channels'] > 0:
            api = sd.query_hostapis(dev['hostapi'])['name']
            if "MME" in api:
                print(f"Microphone found: {dev['name']} (ID: {i})")
                return i
    
    print("Warning: Arctis microphone not found, falling back to default device.")
    return None

def process_audio(audio_array):
    """Pad or truncate the audio array to the target length and convert to MelSpectrogram."""
    target_len = 64 * 512
    if len(audio_array) > target_len:
        audio_array = audio_array[:target_len]
    else:
        padding = target_len - len(audio_array)
        audio_array = np.pad(audio_array, (0, padding), mode='constant')

    wav = torch.tensor(audio_array).float().unsqueeze(0)
    spec = T.MelSpectrogram(sample_rate=FS, n_fft=1024, hop_length=512, n_mels=64)(wav)
    spec = T.AmplitudeToDB()(spec)
    
    return spec.unsqueeze(0).to(DEVICE)

def main():
    print("Loading system...")
    model = SpeakerEncoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    try:
        ref_embedding = torch.load(REF_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: Reference embedding file '{REF_PATH}' not found.")
        return

    mic_id = get_input_device_id()

    print("\n" + "="*40)
    print("   VOICE AUTHENTICATION SYSTEM")
    print("="*40)
    
    input("Press ENTER to start speaking...") 
    print("RECORDING... SPEAK NOW!")
    
    recording = sd.rec(
        int(DURATION * FS), 
        samplerate=FS, 
        channels=1, 
        device=mic_id
    )
    sd.wait()
    print("Stopped. Analyzing...")

    max_vol = np.max(np.abs(recording))
    if max_vol < 0.001:
        print(f"Silence detected (Max vol: {max_vol:.6f}). Please check your microphone mute button.")
        return

    audio_flat = recording.flatten()
    
    with torch.no_grad():
        spec = process_audio(audio_flat)
        new_embedding = model(spec)
        similarity = torch.nn.functional.cosine_similarity(new_embedding, ref_embedding).item()

    print(f"\nSimilarity Score: {similarity:.4f}")
    
    if similarity > THRESHOLD:
        print("SUCCESS: Voice recognized. Access granted.")
    else:
        print("DENIED: Voice not recognized.")
        
    print("-" * 30)

if __name__ == "__main__":
    while True:
        main()
        if input("Try again? (y/n): ").strip().lower() != 'y': 
            break