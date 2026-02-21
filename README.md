# Voice Authentication System

## Description

Deep-learning speaker authentication. Maps Mel-spectrograms to 128-D L2-normalized embeddings. Live audio is captured with `sounddevice` and compared to a saved reference by cosine similarity.

## Architecture & Approach

- Input: Mel-spectrograms (preprocessing in `src/slice_audio.py` using `librosa`).
- Feature extractor: custom CNN that outputs a 128-dimensional embedding.
- Embeddings: L2-normalized to unit length.
- Loss: `TripletMarginLoss` (anchor, positive, negative).
- Training: triplet sampling and hard negative mining.
- Inference: real-time capture → embedding → cosine similarity against reference.

## Solving Shortcut Learning / Data Bias

- Observed issue: model memorized microphone-specific noise (Arctis Nova 7) instead of voice formants.
- Remedy: Hard Negative Mining. Record negatives (other speakers) on the same microphone.
- Effect: forces the network to ignore mic-specific artifacts and learn speaker features.

## Tech Stack

- PyTorch
- torchaudio
- librosa
- sounddevice

## Usage / How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Record a reference voice (captured by `src/record_dataset.py`):

```bash
python src/record_dataset.py
```

Create and save a reference embedding:

```bash
python src/create_reference.py --input data/my_voice/arctis_voice_1 --output my_voice_embedding.pt
```

Train the model (adjust hyperparameters in `src/train.py`):

```bash
python src/train.py --data_dir dev-clean --output voice_auth_model.pth --epochs 100
```

Run real-time verification:

```bash
python src/verify_me.py --reference my_voice_embedding.pt --threshold 0.7
```

Files of interest:

- `src/model.py` — model and embedding logic
- `src/train.py` — training loop and triplet handling
- `src/record_dataset.py` — microphone capture utilities
- `src/create_reference.py` — generate and save reference embedding
- `src/verify_me.py` — real-time inference and similarity check
