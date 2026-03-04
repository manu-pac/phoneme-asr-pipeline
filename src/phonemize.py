# =============================================================================
# Stage 1: Text → Phonemes
# Reads clean.jsonl for each language, runs espeak-ng on each ref_text
# to fill in ref_phon, and writes clean_phonemized.jsonl.
# =============================================================================

import json
import subprocess
from pathlib import Path
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
MANIFEST_DIR_BASE = Path(params["data"]["manifest_dir"])

def phonemize(text: str, lang: str) -> str:
    result = subprocess.run(
        ["espeak-ng", "-v", lang, "-q", "--ipa", text],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    return result.stdout.strip()

for lang in LANGUAGES:
    MANIFEST_DIR = MANIFEST_DIR_BASE / lang
    INPUT_MANIFEST = MANIFEST_DIR / "clean.jsonl"
    OUTPUT_MANIFEST = MANIFEST_DIR / "clean_phonemized.jsonl"

    print(f"\n=== Phonemizing language: {lang} ===")
    records = []

    with open(INPUT_MANIFEST, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ref_phon = phonemize(record["ref_text"], lang)
            record["ref_phon"] = ref_phon
            records.append(record)
            print(f"  {record['utt_id']}: {ref_phon[:50]}...")

    tmp_path = Path(str(OUTPUT_MANIFEST) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(OUTPUT_MANIFEST)
    print(f"  Written: {OUTPUT_MANIFEST}")

print("\nAll languages phonemized!")