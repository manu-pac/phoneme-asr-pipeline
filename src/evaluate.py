# =============================================================================
# Stage 4: Evaluate — Compute Phoneme Error Rate (PER)
# Reads prediction manifests for each language, computes PER,
# and writes metrics/per_{lang}.json.
# =============================================================================

import json
import re
from pathlib import Path
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
SNR_LEVELS = params["snr_levels"]
MANIFEST_DIR_BASE = Path(params["data"]["manifest_dir"])
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

def normalize_phonemes(phon_str: str) -> list:
    phon_str = phon_str.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    phon_str = phon_str.replace(" ", "")
    return list(phon_str)

def compute_per(ref: str, pred: str) -> float:
    ref_tokens = normalize_phonemes(ref)
    pred_tokens = normalize_phonemes(pred)
    N = len(ref_tokens)
    if N == 0:
        return 0.0
    dp = [[0] * (N + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(len(pred_tokens) + 1):
        dp[i][0] = i
    for j in range(N + 1):
        dp[0][j] = j
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, N + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len(pred_tokens)][N] / N

def evaluate_manifest(manifest_path: Path) -> float:
    per_scores = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ref = record.get("ref_phon", "")
            pred = record.get("pred_phon", "")
            if ref and pred:
                per_scores.append(compute_per(ref, pred))
    return sum(per_scores) / len(per_scores) if per_scores else 0.0

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
    tmp_path.rename(metrics_path)
    print(f"  Metrics written: {metrics_path}")

print("\nAll languages evaluated!")