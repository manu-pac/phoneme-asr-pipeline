import json
import os
import torch
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import librosa
import yaml

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
SNR_LEVELS = params["snr_levels"]
MANIFEST_DIR_BASE = Path(params["data"]["manifest_dir"])
MODEL_NAME = params["model"]["name"]
TARGET_SR = params["model"]["sample_rate"]

print(f"Loading model: {MODEL_NAME}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")

def predict_phonemes(wav_path: str) -> str:
    signal, sr = sf.read(wav_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
    inputs = processor(signal, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

for lang in LANGUAGES:
    MANIFEST_DIR = MANIFEST_DIR_BASE / lang
    print(f"\n=== Running ASR for language: {lang} ===")

    manifests = [MANIFEST_DIR / "clean_phonemized.jsonl"]
    for snr in SNR_LEVELS:
        manifests.append(MANIFEST_DIR / f"noisy_snr_{snr}.jsonl")

    for manifest_path in manifests:
        print(f"  Processing {manifest_path.name}")
        final_path = MANIFEST_DIR / f"pred_{manifest_path.stem}.jsonl"
        tmp_path = Path(str(final_path) + ".tmp")

        # Process one record at a time — no full list in memory
        with open(manifest_path, encoding="utf-8") as fin, \
             open(tmp_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                record = json.loads(line)
                record["pred_phon"] = predict_phonemes(record["wav_path"])
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"    [{i+1}] {record['utt_id']}: {record['pred_phon'][:40]}...")

        tmp_path.replace(final_path)
        print(f"    Written: {final_path}")

print("\nAll languages done!")