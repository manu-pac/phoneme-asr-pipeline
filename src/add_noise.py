# =============================================================================
# Stage 2: Add Noise
# Reads clean_phonemized.jsonl, creates noisy versions of each audio file
# at each SNR level defined in params.yaml, and writes one manifest per
# SNR level. Uses the exact noise functions provided by the lab.
# =============================================================================

import json
import numpy as np
import soundfile as sf
from pathlib import Path
import yaml

# =============================================================================
# Load parameters from params.yaml
# =============================================================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["languages"][0]              # e.g. "en"
SNR_LEVELS = params["snr_levels"]          # list of SNR values in dB
SEED = params["seed"]                      # random seed for reproducibility
MANIFEST_DIR = Path(params["data"]["manifest_dir"]) / LANG
NOISY_DIR = Path(params["data"]["noisy_dir"]) / LANG  # data/noisy/en

INPUT_MANIFEST = MANIFEST_DIR / "clean_phonemized.jsonl"

# Create noisy audio output folder
NOISY_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Noise functions provided by the lab
# add_noise: pure function — takes signal, returns noisy signal
# add_noise_to_file: handles file I/O, delegates to add_noise
# =============================================================================
def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Estimate average power of the clean signal
    signal_power = np.mean(signal ** 2)
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    # Compute required noise power to achieve the requested SNR
    noise_power = signal_power / snr_linear
    # Generate white Gaussian noise with the required power
    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(
    input_wav: str,
    output_wav: str,
    snr_db: float,
    seed: int | None = None,
) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)

# =============================================================================
# Read input manifest
# =============================================================================
print(f"Reading manifest: {INPUT_MANIFEST}")
records = []
with open(INPUT_MANIFEST, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# =============================================================================
# For each SNR level, create noisy audio files and write a manifest
# =============================================================================
for snr_db in SNR_LEVELS:
    print(f"\nProcessing SNR = {snr_db} dB")

    # Folder for this SNR level e.g. data/noisy/en/snr_10/
    snr_dir = NOISY_DIR / f"snr_{snr_db}"
    snr_dir.mkdir(parents=True, exist_ok=True)

    noisy_records = []

    for record in records:
        # Build output path for the noisy wav file
        original_stem = Path(record["wav_path"]).stem
        noisy_wav_path = snr_dir / f"{original_stem}.wav"

        # Add noise to the clean wav file
        add_noise_to_file(
            input_wav=record["wav_path"],
            output_wav=str(noisy_wav_path),
            snr_db=snr_db,
            seed=SEED   # fixed seed ensures reproducibility
        )

        # Build the noisy manifest record — same as clean but with
        # updated wav_path and snr_db fields
        noisy_record = dict(record)  # copy all fields from clean record
        noisy_record["wav_path"] = str(noisy_wav_path).replace("\\", "/")
        noisy_record["snr_db"] = snr_db

        noisy_records.append(noisy_record)
        print(f"  {noisy_wav_path.name}")

    # Write manifest for this SNR level atomically
    final_path = MANIFEST_DIR / f"noisy_snr_{snr_db}.jsonl"
    tmp_path = Path(str(final_path) + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in noisy_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tmp_path.rename(final_path)
    print(f"  Manifest written to {final_path}")

print(f"\nDone! Created {len(SNR_LEVELS)} noisy manifests.")