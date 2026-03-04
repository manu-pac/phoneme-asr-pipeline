# =============================================================================
# Stage 2: Add Noise
# Reads clean_phonemized.jsonl for each language, creates noisy versions
# at each SNR level, and writes one manifest per SNR level.
# =============================================================================

import json
import numpy as np
import soundfile as sf
from pathlib import Path
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
SNR_LEVELS = params["snr_levels"]
SEED = params["seed"]
MANIFEST_DIR_BASE = Path(params["data"]["manifest_dir"])
NOISY_DIR_BASE = Path(params["data"]["noisy_dir"])

def add_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def add_noise_to_file(input_wav, output_wav, snr_db, seed=None):
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)

for lang in LANGUAGES:
    MANIFEST_DIR = MANIFEST_DIR_BASE / lang
    NOISY_DIR = NOISY_DIR_BASE / lang
    INPUT_MANIFEST = MANIFEST_DIR / "clean_phonemized.jsonl"

    print(f"\n=== Adding noise for language: {lang} ===")

    records = []
    with open(INPUT_MANIFEST, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    for snr_db in SNR_LEVELS:
        print(f"  SNR = {snr_db} dB")
        snr_dir = NOISY_DIR / f"snr_{snr_db}"
        snr_dir.mkdir(parents=True, exist_ok=True)

        noisy_records = []
        for record in records:
            stem = Path(record["wav_path"]).stem
            noisy_wav = snr_dir / f"{stem}.wav"
            add_noise_to_file(record["wav_path"], str(noisy_wav), snr_db, seed=SEED)
            noisy_record = dict(record)
            noisy_record["wav_path"] = str(noisy_wav).replace("\\", "/")
            noisy_record["snr_db"] = snr_db
            noisy_records.append(noisy_record)

        final_path = MANIFEST_DIR / f"noisy_snr_{snr_db}.jsonl"
        tmp_path = Path(str(final_path) + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in noisy_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path.replace(final_path)
        print(f"    Manifest written: {final_path}")

print("\nAll languages done!")