"""Generate non-destructive exclude list from classification results.

Reads gt_classifications.json and outputs filtered_samples.json
that PairedDataset can load to exclude Mode B samples.

Run: python src/data_processing/exp_01/generate_exclude.py
"""

import json
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).parent
FILTERING_DIR = EXP_DIR.parent
CONFIG_PATH = EXP_DIR / "config.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"


def main():
    classifications_path = OUTPUT_DIR / "gt_classifications.json"
    if not classifications_path.exists():
        print(f"ERROR: {classifications_path} not found.")
        print("Run classify.py first.")
        sys.exit(1)

    with open(classifications_path) as f:
        data = json.load(f)

    excluded = []
    reasons_map = {}
    for sample in data["samples"]:
        if sample["mode"] == "B":
            excluded.append(sample["filename"])
            reasons_map[sample["filename"]] = sample["reasons"]

    output = {
        "excluded": sorted(excluded),
        "reasons": reasons_map,
        "config_used": {
            "lap_var_threshold": data["thresholds"]["lap_var_blur"],
            "noise_est_threshold": data["thresholds"]["noise_est_noisy"],
            "local_var_threshold": data["thresholds"]["local_var_flat"],
        },
        "total_excluded": len(excluded),
        "total_remaining": data["summary"]["total"] - len(excluded),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "filtered_samples.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Excluded: {len(excluded)} samples")
    print(f"Remaining: {output['total_remaining']} samples")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
