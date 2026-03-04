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
    tmp_path = Path(str(OUTPUT_MANIFEST) + ".tmp")

    print(f"\n=== Phonemizing language: {lang} ===")

    # Process one record at a time — no full list in memory
    with open(INPUT_MANIFEST, encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            record["ref_phon"] = phonemize(record["ref_text"], lang)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {record['utt_id']}: {record['ref_phon'][:50]}...")

    tmp_path.replace(OUTPUT_MANIFEST)
    print(f"  Written: {OUTPUT_MANIFEST}")

print("\nAll languages phonemized!")