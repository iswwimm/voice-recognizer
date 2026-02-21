import os
import time

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

OUTPUT_FOLDER = "data/my_voices/sliced_1"
FS = 16000 
DURATION = 5 
NUM_SAMPLES = 100

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_mic_id():
    """Locate the specific Arctis headset using the MME host API."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "Arctis" in dev['name'] and dev['max_input_channels'] > 0:
            api = sd.query_hostapis(dev['hostapi'])['name']
            if "MME" in api:
                return i
    return None 

mic_id = get_mic_id()
if mic_id is None:
    print("Error: Arctis microphone not found. Please check the connection.")
    exit()

print(f"Using microphone ID: {mic_id}")
print(f"Target: Record {NUM_SAMPLES} samples.")
print("Tip: Vary your tone, volume, and speaking rate for better dataset diversity.")

input("Press ENTER to start the recording session")

for i in range(NUM_SAMPLES):
    print(f"\n[{i+1}/{NUM_SAMPLES}] Recording")
    recording = sd.rec(
        int(DURATION * FS), 
        samplerate=FS, 
        channels=1, 
        device=mic_id
    )
    sd.wait()
    
    if np.max(np.abs(recording)) < 0.005:
        print("Silence detected. Recording discarded. Please speak louder.")
    else:
        filename = os.path.join(OUTPUT_FOLDER, f"other_voice_arctis_{i}.wav")
        write(filename, FS, recording)
        print("Saved.")
    
    time.sleep(0.5) 

print("\nRecording session complete.")