import os
from glob import glob

import librosa
import soundfile as sf

INPUT_DIR = "data/other_voices/voices_raw_1"
OUTPUT_DIR = "data/other_voices/sliced_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob(os.path.join(INPUT_DIR, "*.wav"))
print(f"Slicing {len(files)} files...")

idx = 0
for f in files:
    audio, sr = librosa.load(f, sr=16000)
    
    for start in range(0, len(audio), 48000):
        chunk = audio[start:start+48000]
        
        if len(chunk) == 48000:
            output_path = os.path.join(OUTPUT_DIR, f"me_{idx}.wav")
            sf.write(output_path, chunk, 16000)
            idx += 1

print("Processing complete.")