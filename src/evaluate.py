# =============================================================================
# Stage 4: Evaluate — Compute Phoneme Error Rate (PER)
# Reads each prediction manifest, compares pred_phon against ref_phon
# using edit distance, computes PER per utterance and averaged across
# all utterances, and writes a JSON metrics file.
# PER = (Substitutions + Deletions + Insertions) / N
# where N is the number of phonemes in the reference sequence.
# =============================================================================

import json
from pathlib import Path
import yaml
import re

# =============================================================================
# Load parameters from params.yaml
# =============================================================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["languages"][0]
MANIFEST_DIR = Path(params["data"]["manifest_dir"]) / LANG
SNR_LEVELS = params["snr_levels"]

# =============================================================================
# Helper function: compute PER between two phoneme strings
# We treat each space-separated token as one phoneme unit.
# This is equivalent to Word Error Rate but at the phoneme level.
# =============================================================================
def normalize_phonemes(phon_str: str) -> list:
    """
    Normalize a phoneme string into a list of comparable tokens.
    - Remove stress markers (ˈ ˌ) and length markers (ː)
    - Remove word boundaries
    - Split into individual characters (each IPA character = one phoneme)
    """
    # Remove stress and length markers
    phon_str = phon_str.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    # Remove spaces (we compare at character level)
    phon_str = phon_str.replace(" ", "")
    # Split into individual unicode characters
    return list(phon_str)

def compute_per(ref: str, pred: str) -> float:
    ref_tokens = normalize_phonemes(ref)
    pred_tokens = normalize_phonemes(pred)

    N = len(ref_tokens)
    if N == 0:
        return 0.0

    # Dynamic programming edit distance
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
                dp[i][j] = 1 + min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1]
                )

    return dp[len(pred_tokens)][N] / N

# =============================================================================
# Process clean manifest first
# =============================================================================
results = {}  # dict of snr_db -> average PER

def evaluate_manifest(manifest_path: Path) -> float:
    """Evaluate a single manifest and return average PER."""
    per_scores = []

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ref = record.get("ref_phon", "")
            pred = record.get("pred_phon", "")

            if ref and pred:
                per = compute_per(ref, pred)
                per_scores.append(per)

    return sum(per_scores) / len(per_scores) if per_scores else 0.0

# Evaluate clean audio (snr_db = None, we'll use "clean" as key)
clean_manifest = MANIFEST_DIR / "pred_clean_phonemized.jsonl"
clean_per = evaluate_manifest(clean_manifest)
results["clean"] = clean_per
print(f"Clean PER: {clean_per:.4f}")

# Evaluate each noisy manifest
for snr in SNR_LEVELS:
    manifest_path = MANIFEST_DIR / f"pred_noisy_snr_{snr}.jsonl"
    per = evaluate_manifest(manifest_path)
    results[str(snr)] = per
    print(f"SNR {snr:>4} dB  PER: {per:.4f}")

# =============================================================================
# Write metrics to a JSON file
# This file will be declared as a DVC metric so it's tracked over time
# =============================================================================
metrics_dir = Path("metrics")
metrics_dir.mkdir(exist_ok=True)

metrics_path = metrics_dir / f"per_{LANG}.json"
tmp_path = Path(str(metrics_path) + ".tmp")

with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

tmp_path.rename(metrics_path)
print(f"\nMetrics written to {metrics_path}")