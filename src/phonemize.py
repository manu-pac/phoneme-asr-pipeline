# =============================================================================
# Stage 1: Text → Phonemes
# Reads clean.jsonl, runs espeak-ng on each ref_text to fill in ref_phon,
# and writes an updated manifest. This stage transforms raw text transcripts
# into phoneme sequences that will later be used for PER evaluation.
# =============================================================================

import json
import subprocess
from pathlib import Path
import yaml

# =============================================================================
# Load parameters from params.yaml
# =============================================================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["languages"][0]          # e.g. "en"
MANIFEST_DIR = Path(params["data"]["manifest_dir"]) / LANG  # data/manifests/en

INPUT_MANIFEST  = MANIFEST_DIR / "clean.jsonl"
OUTPUT_MANIFEST = MANIFEST_DIR / "clean_phonemized.jsonl"

# =============================================================================
# Helper function: run espeak-ng on a single text string
# Returns the IPA phoneme sequence as a string
# =============================================================================
def phonemize(text: str, lang: str) -> str:
    result = subprocess.run(
        [
            "espeak-ng",
            "-v", lang,        # language code e.g. "en"
            "-q",              # quiet mode — no audio output
            "--ipa",           # output IPA phonemes
            text               # the text to phonemize
        ],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    # Strip leading/trailing whitespace and newlines
    return result.stdout.strip()

# =============================================================================
# Read input manifest, phonemize each record, collect updated records
# =============================================================================
print(f"Phonemizing manifest: {INPUT_MANIFEST}")

records = []

with open(INPUT_MANIFEST, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        # Run espeak-ng on the ref_text field
        ref_phon = phonemize(record["ref_text"], LANG)

        # Fill in the ref_phon field
        record["ref_phon"] = ref_phon

        records.append(record)
        print(f"  {record['utt_id']}: {ref_phon[:50]}...")  # print first 50 chars

# =============================================================================
# Write updated manifest ATOMICALLY
# =============================================================================
tmp_path = Path(str(OUTPUT_MANIFEST) + ".tmp")
final_path = OUTPUT_MANIFEST

with open(tmp_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

tmp_path.rename(final_path)

print(f"\nPhonemized manifest written to {final_path} ({len(records)} utterances)")