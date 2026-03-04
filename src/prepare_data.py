# =============================================================================
# Stage 0: Data Preparation
# Reads audio and transcripts from local CommonVoice datasets and creates
# the initial manifest for each language defined in params.yaml.
# CommonVoice format: .mp3 audio files + test.tsv transcript file
# Audio is converted from .mp3 to .wav using ffmpeg.
# =============================================================================

import json
import hashlib
import subprocess
import soundfile as sf
from pathlib import Path
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
NUM_SAMPLES = 30
CV_VERSION = "cv-corpus-24.0-2025-12-05"

# =============================================================================
# Helper: convert .mp3 to .wav using ffmpeg
# =============================================================================
def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",              # overwrite output if exists
            "-i", mp3_path,    # input file
            "-ar", "16000",    # resample to 16000 Hz (required by wav2vec2)
            "-ac", "1",        # convert to mono
            wav_path           # output file
        ],
        capture_output=True,   # suppress ffmpeg output
        check=True             # raise error if ffmpeg fails
    )

# =============================================================================
# Atomic manifest writer
# =============================================================================
def write_manifest(final_path: Path, records: list) -> None:
    tmp_path = Path(str(final_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(final_path)

# =============================================================================
# Prepare data for a single CommonVoice language
# Works identically for any language — no special cases needed
# =============================================================================
def prepare_language(lang: str) -> None:
    # Paths
    CV_DIR = Path(params["data"]["raw_dir"]) / lang / CV_VERSION / lang
    WAV_DIR = Path(params["data"]["raw_dir"]) / lang / "wav"
    MANIFEST_DIR = Path(params["data"]["manifest_dir"]) / lang
    TSV_PATH = CV_DIR / "test.tsv"
    CLIPS_DIR = CV_DIR / "clips"

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Preparing language: {lang} ===")
    print(f"Reading {TSV_PATH}")

    # Read test.tsv — tab separated, first line is header
    pairs = []  # list of (mp3_filename, transcript)
    with open(TSV_PATH, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        path_idx = header.index("path")
        sentence_idx = header.index("sentence")

        for line in f:
            cols = line.strip().split("\t")
            if len(cols) <= max(path_idx, sentence_idx):
                continue
            mp3_name = cols[path_idx]
            transcript = cols[sentence_idx]
            pairs.append((mp3_name, transcript))

            if len(pairs) >= NUM_SAMPLES:
                break

    print(f"  Found {len(pairs)} utterances to process")

    records = []

    for mp3_name, transcript in pairs:
        stem = mp3_name.replace(".mp3", "")
        utt_id = f"{lang}_{stem}"
        mp3_path = CLIPS_DIR / mp3_name
        wav_path = WAV_DIR / f"{stem}.wav"

        # Convert .mp3 to .wav (16000 Hz mono)
        convert_mp3_to_wav(str(mp3_path), str(wav_path))

        # Read converted wav to get duration and sr
        audio_array, sr = sf.read(str(wav_path))
        md5 = hashlib.md5(wav_path.read_bytes()).hexdigest()

        records.append({
            "utt_id": utt_id,
            "lang": lang,
            "wav_path": str(wav_path).replace("\\", "/"),
            "ref_text": transcript,
            "ref_phon": None,       # filled by Stage 1
            "sr": sr,
            "duration_s": round(len(audio_array) / sr, 2),
            "snr_db": None,         # filled by Stage 2
            "audio_md5": md5
        })
        print(f"  Processed {wav_path.name}")

    write_manifest(MANIFEST_DIR / "clean.jsonl", records)
    print(f"  Manifest written: {MANIFEST_DIR}/clean.jsonl ({len(records)} utterances)")

# =============================================================================
# Run for all languages in params.yaml
# =============================================================================
for lang in LANGUAGES:
    prepare_language(lang)

print("\nAll languages done!")