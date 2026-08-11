"""Generate separate exclude lists for noise, blur, and combined (exp_02).

Reads classifications from exp_02/noise/ and exp_02/blur/,
writes exclude_list.json in each folder plus a combined one at exp_02/.

Run: python src/data_processing/generate_exclude_exp02.py
"""

import json
import sys
from pathlib import Path

EXP02_DIR = Path(__file__).parent / "exp_02"


def generate(dimension):
    cls_path = EXP02_DIR / dimension / "classifications.json"
    if not cls_path.exists():
        print(f"  ERROR: {cls_path} not found. Run classify_exp02.py first.")
        return None

    with open(cls_path) as f:
        data = json.load(f)

    excluded = [s["filename"] for s in data["samples"] if s["flagged"]]

    output = {
        "dimension": dimension,
        "threshold": data["threshold"],
        "excluded": sorted(excluded),
        "total_excluded": len(excluded),
        "total_remaining": data["summary"]["total"] - len(excluded),
    }

    out_path = EXP02_DIR / dimension / "exclude_list.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  {dimension}: excluded {len(excluded)}, "
          f"remaining {output['total_remaining']}")
    print(f"  Saved {out_path}")
    return set(excluded)


def main():
    print("=== Generate Exclude Lists (exp_02) ===\n")
    noise_ex = generate("noise")
    blur_ex = generate("blur")

    if noise_ex is not None and blur_ex is not None:
        both = noise_ex & blur_ex
        only_noise = noise_ex - blur_ex
        only_blur = blur_ex - noise_ex
        union = noise_ex | blur_ex

        print(f"\n--- Overlap ---")
        print(f"  Noise only:  {len(only_noise)}")
        print(f"  Blur only:   {len(only_blur)}")
        print(f"  Both:        {len(both)}")
        print(f"  Union:       {len(union)}")

        # Combined exclude list
        with open(EXP02_DIR / "noise" / "exclude_list.json") as f:
            noise_data = json.load(f)
        with open(EXP02_DIR / "blur" / "exclude_list.json") as f:
            blur_data = json.load(f)

        combined = {
            "dimension": "combined",
            "noise_threshold": noise_data["threshold"],
            "blur_threshold": blur_data["threshold"],
            "excluded": sorted(union),
            "total_excluded": len(union),
            "total_remaining": 2560 - len(union),
            "breakdown": {
                "noise_only": len(only_noise),
                "blur_only": len(only_blur),
                "both": len(both),
            },
        }

        combined_path = EXP02_DIR / "exclude_list.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Combined: {len(union)} excluded, {2560 - len(union)} remaining")
        print(f"  Saved {combined_path}")


if __name__ == "__main__":
    main()
