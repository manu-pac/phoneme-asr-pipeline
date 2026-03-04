import json
from pathlib import Path
import yaml
from jiwer import wer

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
SNR_LEVELS = params["snr_levels"]
MANIFEST_DIR_BASE = Path(params["data"]["manifest_dir"])
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

def normalize_phonemes(phon_str: str) -> str:
    """
    Normalize a phoneme string for comparison.
    - Remove stress and length markers
    - Split into space-separated individual characters
      so jiwer treats each IPA character as one token
    """
    phon_str = phon_str.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    phon_str = phon_str.replace(" ", "")
    # Return as space-separated characters so jiwer computes
    # token-level edit distance (equivalent to PER)
    return " ".join(list(phon_str))

def compute_per(ref: str, pred: str) -> float:
    ref_normalized = normalize_phonemes(ref)
    pred_normalized = normalize_phonemes(pred)
    if not ref_normalized:
        return 0.0
    # jiwer's wer computes (S+D+I)/N — identical to PER formula
    return wer(ref_normalized, pred_normalized)

def evaluate_manifest(manifest_path: Path) -> float:
    # Process one record at a time — no full list in memory
    total_per = 0.0
    count = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ref = record.get("ref_phon", "")
            pred = record.get("pred_phon", "")
            if ref and pred:
                total_per += compute_per(ref, pred)
                count += 1
    return total_per / count if count > 0 else 0.0

for lang in LANGUAGES:
    MANIFEST_DIR = MANIFEST_DIR_BASE / lang
    print(f"\n=== Evaluating language: {lang} ===")

    results = {}

    clean_per = evaluate_manifest(MANIFEST_DIR / "pred_clean_phonemized.jsonl")
    results["clean"] = clean_per
    print(f"  Clean PER: {clean_per:.4f}")

    for snr in SNR_LEVELS:
        per = evaluate_manifest(MANIFEST_DIR / f"pred_noisy_snr_{snr}.jsonl")
        results[str(snr)] = per
        print(f"  SNR {snr:>4} dB  PER: {per:.4f}")

    metrics_path = METRICS_DIR / f"per_{lang}.json"
    tmp_path = Path(str(metrics_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    tmp_path.replace(metrics_path)
    print(f"  Metrics written: {metrics_path}")

print("\nAll languages evaluated!")