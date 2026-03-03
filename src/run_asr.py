# =============================================================================
# Stage 3: Run ASR Model
# Reads each manifest (clean + all noisy), runs the wav2vec2 phoneme
# recogniser on each audio file, and writes prediction manifests.
# Model: facebook/wav2vec2-lv-60-espeak-cv-ft
# =============================================================================

import json
import torch
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np
import librosa
import yaml
import os

# =============================================================================
# Point phonemizer to the espeak-ng shared library (required on Windows)
# =============================================================================
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

# =============================================================================
# Load parameters from params.yaml
# =============================================================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["languages"][0]
MANIFEST_DIR = Path(params["data"]["manifest_dir"]) / LANG
MODEL_NAME = params["model"]["name"]
TARGET_SR = params["model"]["sample_rate"]  # must be 16000 Hz

# =============================================================================
# Load the pre-trained wav2vec2 model and processor from HuggingFace
# This will download the model the first time it runs (~1GB)
# =============================================================================
print(f"Loading model: {MODEL_NAME}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()  # set to evaluation mode — disables dropout etc.
print("Model loaded.")

# =============================================================================
# Helper function: run inference on a single wav file
# Returns predicted phoneme sequence as a string
# =============================================================================
def predict_phonemes(wav_path: str) -> str:
    # Read the audio file
    signal, sr = sf.read(wav_path)

    # Ensure mono audio — the model only accepts single-channel audio
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Resample if necessary — wav2vec2 requires exactly 16000 Hz
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Preprocess the signal into the format the model expects
    inputs = processor(
        signal,
        sampling_rate=TARGET_SR,
        return_tensors="pt"  # return PyTorch tensors
    )

    # Run inference — no gradient computation needed for prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode the model output into a phoneme sequence
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_phonemes = processor.batch_decode(predicted_ids)[0]

    return predicted_phonemes

# =============================================================================
# Collect all manifests to process: clean + all noisy SNR levels
# =============================================================================
manifests_to_process = [MANIFEST_DIR / "clean_phonemized.jsonl"]

for snr in params["snr_levels"]:
    manifests_to_process.append(MANIFEST_DIR / f"noisy_snr_{snr}.jsonl")

# =============================================================================
# Process each manifest
# =============================================================================
for manifest_path in manifests_to_process:
    print(f"\nProcessing {manifest_path.name}")

    records = []
    with open(manifest_path, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        record = json.loads(line)

        # Run the model on this audio file
        pred_phon = predict_phonemes(record["wav_path"])

        # Add prediction to the record
        record["pred_phon"] = pred_phon

        records.append(record)
        print(f"  [{i+1}/{len(lines)}] {record['utt_id']}: {pred_phon[:50]}...")

    # Write prediction manifest atomically
    stem = manifest_path.stem  # e.g. "clean_phonemized" or "noisy_snr_10"
    final_path = MANIFEST_DIR / f"pred_{stem}.jsonl"
    tmp_path = Path(str(final_path) + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tmp_path.rename(final_path)
    print(f"  Predictions written to {final_path}")

print(f"\nDone! Processed {len(manifests_to_process)} manifests.")