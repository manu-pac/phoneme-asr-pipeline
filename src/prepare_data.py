# =============================================================================
# Stage 0: Data Preparation
# Reads audio and transcripts from a local LibriSpeech folder and creates
# the initial manifest. This is the entry point of the pipeline — it produces
# clean.jsonl which all subsequent stages depend on.
# =============================================================================

import json        # for writing .jsonl manifest (one JSON object per line)
import hashlib     # for computing audio_md5 checksum (required by the lab)
import soundfile as sf  # for reading .flac and writing .wav files
from pathlib import Path  # for cross-platform file path handling
import yaml        # for reading params.yaml

# =============================================================================
# Load parameters from params.yaml
# This is what allows the pipeline to work for any language without
# changing the code — we just change params.yaml
# =============================================================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["languages"][0]          # e.g. "en"
RAW_DIR = Path(params["data"]["raw_dir"]) / LANG / "wav"    # data/raw/en/wav
MANIFEST_DIR = Path(params["data"]["manifest_dir"]) / LANG  # data/manifests/en
NUM_SAMPLES = 30  # how many clips to use — small enough to work with easily

# Where LibriSpeech was extracted
LIBRISPEECH_DIR = Path(params["data"]["raw_dir"]) / LANG / "LibriSpeech" / "test-clean"

# =============================================================================
# Create output folders if they don't exist yet
# =============================================================================
RAW_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Read LibriSpeech folder structure
# Each speaker folder contains audio .flac files and a .trans.txt transcript
# file. We walk through all of them and collect (audio_path, transcript) pairs.
# =============================================================================
print(f"Reading LibriSpeech from {LIBRISPEECH_DIR}")

pairs = []  # list of (flac_path, transcript_text) tuples

for trans_file in sorted(LIBRISPEECH_DIR.rglob("*.trans.txt")):
    # Each line in the transcript file is: UTTERANCE_ID transcript text
    with open(trans_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first space: "1089-134686-0000 SENTENCE HERE"
            utt_id_raw, transcript = line.split(" ", 1)
            # Find the corresponding .flac file
            flac_path = trans_file.parent / f"{utt_id_raw}.flac"
            if flac_path.exists():
                pairs.append((flac_path, transcript))

    # Stop once we have enough
    if len(pairs) >= NUM_SAMPLES:
        break

# Take only the first NUM_SAMPLES
pairs = pairs[:NUM_SAMPLES]
print(f"  Found {len(pairs)} utterances")

# =============================================================================
# Process each audio clip and build the list of manifest records
# =============================================================================
records = []

for flac_path, transcript in pairs:
    stem = flac_path.stem  # e.g. "1089-134686-0000"
    utt_id = f"{LANG}_{stem}"  # follows lab's recommendation: {lang}_{stem}
    wav_path = RAW_DIR / f"{stem}.wav"

    # Read .flac and save as .wav
    # The pipeline and wav2vec2 model both require .wav format
    audio_array, sr = sf.read(str(flac_path))
    sf.write(str(wav_path), audio_array, sr)

    # Compute md5 checksum of the saved .wav file
    # The lab requires this for traceability
    md5 = hashlib.md5(wav_path.read_bytes()).hexdigest()

    # Build the manifest record with all fields required by the lab
    record = {
        "utt_id": utt_id,
        "lang": LANG,
        "wav_path": str(wav_path).replace("\\", "/"),  # forward slashes for cross-platform
        "ref_text": transcript,   # the raw text transcript
        "ref_phon": None,         # filled in by Stage 1 (espeak-ng)
        "sr": sr,                 # sampling rate in Hz
        "duration_s": round(len(audio_array) / sr, 2),  # duration in seconds
        "snr_db": None,           # null for clean audio (no noise added yet)
        "audio_md5": md5          # checksum for traceability
    }
    records.append(record)
    print(f"  Processed {wav_path.name}")

# =============================================================================
# Write the manifest ATOMICALLY
# Write to a temp file first, then rename to final path only when everything
# is done — if the script crashes halfway, no incomplete manifest exists
# =============================================================================
tmp_path = MANIFEST_DIR / "clean.jsonl.tmp"
final_path = MANIFEST_DIR / "clean.jsonl"

with open(tmp_path, "w", encoding="utf-8") as f:
    for r in records:
        # Each line is one valid JSON object — this is the .jsonl format
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Rename temp file to final path — this is the atomic step
tmp_path.rename(final_path)

print(f"\nManifest written to {final_path} ({len(records)} utterances)")