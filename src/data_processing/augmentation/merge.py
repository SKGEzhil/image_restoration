"""Merge synthetic data into the main dataset.

Copies synthetic GT/NoisyLR images into src/data/train/ and updates
split.json to include the new samples.

Usage:
    python merge.py
    python merge.py --dry-run  # preview without copying
    python merge.py --synthetic-dir outputs/synthetic
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def load_split(split_path: Path) -> dict:
    """Load split.json."""
    with open(split_path) as f:
        return json.load(f)


def save_split(split_path: Path, split_data: dict):
    """Save split.json with backup."""
    backup = split_path.with_suffix(".json.bak")
    shutil.copy2(split_path, backup)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)
    return backup


def merge(
    synthetic_dir: Path,
    data_dir: Path,
    split_path: Path,
    dry_run: bool = False,
) -> dict:
    """Merge synthetic data into main dataset.

    Returns summary dict.
    """
    synth_gt_dir = synthetic_dir / "GT"
    synth_lr_dir = synthetic_dir / "NoisyLR"

    # Validate synthetic dir
    if not synth_gt_dir.exists():
        raise FileNotFoundError(f"Synthetic GT dir not found: {synth_gt_dir}")
    if not synth_lr_dir.exists():
        raise FileNotFoundError(f"Synthetic NoisyLR dir not found: {synth_lr_dir}")

    # Get synthetic files
    synth_gt_files = sorted(p.name for p in synth_gt_dir.glob("*.npy"))
    synth_lr_files = sorted(p.name for p in synth_lr_dir.glob("*.npy"))

    assert synth_gt_files == synth_lr_files, "Synthetic GT/NoisyLR filename mismatch"
    print(f"Found {len(synth_gt_files)} synthetic pairs")

    # Load split
    split_data = load_split(split_path)
    existing_train = set(split_data["train"]["samples"])

    # Check for conflicts
    conflicts = [f for f in synth_gt_files if f in existing_train]
    if conflicts:
        print(f"\nWARNING: {len(conflicts)} filename conflicts found:")
        for c in conflicts[:10]:
            print(f"  {c}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")

        if not dry_run:
            response = input("\nContinue anyway? These files will be OVERWRITTEN. (y/N): ")
            if response.lower() != "y":
                print("Aborted.")
                return {"status": "aborted", "reason": "filename conflicts"}

    # Target directories
    target_gt_dir = data_dir / "train" / "GT"
    target_lr_dir = data_dir / "train" / "NoisyLR"

    # Copy files
    copied = 0
    skipped = 0

    for name in synth_gt_files:
        src_gt = synth_gt_dir / name
        src_lr = synth_lr_dir / name
        dst_gt = target_gt_dir / name
        dst_lr = target_lr_dir / name

        if dry_run:
            print(f"  [DRY RUN] Would copy: {name}")
            copied += 1
        else:
            shutil.copy2(src_gt, dst_gt)
            shutil.copy2(src_lr, dst_lr)
            copied += 1

    # Update split.json
    new_samples = [f for f in synth_gt_files if f not in existing_train]
    old_samples = [f for f in synth_gt_files if f in existing_train]

    if not dry_run:
        split_data["train"]["samples"].extend(new_samples)
        split_data["train"]["samples"].sort()
        split_data["train"]["count"] = len(split_data["train"]["samples"])
        split_data["total"] = sum(
            split_data[s]["count"]
            for s in ["train", "val", "test"]
            if s in split_data
        )
        backup_path = save_split(split_path, split_data)
        print(f"Backed up split.json to {backup_path}")

    summary = {
        "status": "dry_run" if dry_run else "completed",
        "timestamp": datetime.now().isoformat(),
        "synthetic_dir": str(synthetic_dir),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "total_synthetic": len(synth_gt_files),
        "copied": copied,
        "new_added": len(new_samples),
        "overwritten": len(old_samples),
        "old_train_count": split_data["train"]["count"] - len(new_samples) if not dry_run else None,
        "new_train_count": split_data["train"]["count"] if not dry_run else None,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge synthetic data into main dataset")
    parser.add_argument("--synthetic-dir", type=str, default=None,
                        help="Path to outputs/synthetic (default: outputs/synthetic)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to src/data (default: ../../../data)")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split.json (default: data_dir/split.json)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    synthetic_dir = Path(args.synthetic_dir) if args.synthetic_dir else base_dir / "outputs" / "synthetic"

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        # Read from config
        config_path = base_dir / "augmentation_config.yaml"
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        data_dir = (base_dir / config["data_dir"]).resolve()

    split_path = Path(args.split_file) if args.split_file else data_dir / "split.json"

    print(f"Synthetic dir: {synthetic_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Split file: {split_path}")
    print()

    summary = merge(synthetic_dir, data_dir, split_path, dry_run=args.dry_run)

    print(f"\n{'='*50}")
    print(f"Status: {summary['status']}")
    if not args.dry_run:
        print(f"Old train count: {summary['old_train_count']}")
        print(f"New train count: {summary['new_train_count']}")
    print(f"Copied: {summary['copied']} files")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
