# =============================================================================
# Stage 5: Plot — Performance vs Noise
# Reads the metrics file and plots PER as a function of SNR level.
# Produces one curve per language and a cross-language mean curve.
# =============================================================================

import json
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# =============================================================================
# Load parameters from params.yaml
# =============================================================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGUAGES = params["languages"]
SNR_LEVELS = params["snr_levels"]
METRICS_DIR = Path("metrics")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Load PER metrics for each language
# =============================================================================
all_per = {}  # dict of lang -> list of PER values (one per SNR level)

for lang in LANGUAGES:
    metrics_path = METRICS_DIR / f"per_{lang}.json"
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    # Extract PER values in SNR order (excluding clean)
    per_values = [metrics[str(snr)] for snr in SNR_LEVELS]
    all_per[lang] = per_values
    print(f"Loaded metrics for {lang}")

# =============================================================================
# Plot PER vs SNR for each language + cross-language mean
# =============================================================================
plt.figure(figsize=(10, 6))

for lang, per_values in all_per.items():
    plt.plot(SNR_LEVELS, per_values, marker="o", label=lang)

# Compute and plot cross-language mean if more than one language
if len(LANGUAGES) > 1:
    mean_per = [
        sum(all_per[lang][i] for lang in LANGUAGES) / len(LANGUAGES)
        for i in range(len(SNR_LEVELS))
    ]
    plt.plot(SNR_LEVELS, mean_per, marker="s", linestyle="--",
             color="black", label="mean")

# Formatting
plt.xlabel("SNR (dB)")
plt.ylabel("Phoneme Error Rate (PER)")
plt.title("ASR Robustness to Noise")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # left = most noise, right = cleanest

# Save the plot atomically
tmp_path = PLOTS_DIR / "per_vs_snr_tmp.png"
final_path = PLOTS_DIR / "per_vs_snr.png"
plt.savefig(str(tmp_path), dpi=150, bbox_inches="tight")
tmp_path.rename(final_path)

print(f"Plot saved to {final_path}")