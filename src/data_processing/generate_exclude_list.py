"""Generate non-destructive exclude list from classification results.

Reads gt_classifications.json and outputs filtered_samples.json
that PairedDataset can load to exclude Mode B samples.

Run: python src/data_processing/generate_exclude_list.py
"""

import json
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CLASSIFICATIONS_JSON = Path(__file__).parent / ".." / "data" / "train" / "gt_classifications.json"
OUTPUT_JSON = Path(__file__).parent / ".." / "data" / "train" / "filtered_samples.json"


def main():
    if not CLASSIFICATIONS_JSON.exists():
        print(f"ERROR: {CLASSIFICATIONS_JSON} not found.")
        print("Run classify_gt.py first.")
        sys.exit(1)

    with open(CLASSIFICATIONS_JSON) as f:
        data = json.load(f)

    with open(CONFIG_PATH) as f:
        import yaml
        cfg = yaml.safe_load(f)

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

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Excluded: {len(excluded)} samples")
    print(f"Remaining: {output['total_remaining']} samples")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
