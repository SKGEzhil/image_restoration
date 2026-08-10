import json
import os
import shutil
from pathlib import Path

import numpy as np

RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def main(data_dir: str | None = None) -> None:
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent / "src" / "data")
    root = Path(data_dir)
    src = root / "train"
    gt_files = sorted(p.name for p in (src / "GT").glob("*.npy"))
    noisy_files = sorted(p.name for p in (src / "NoisyLR").glob("*.npy"))
    assert gt_files == noisy_files, "GT and NoisyLR filenames do not match"

    total = len(gt_files)
    idx = np.random.default_rng(SEED).permutation(total)

    n_train = int(round(total * RATIOS[0]))
    n_val = int(round(total * RATIOS[1]))
    n_test = total - n_train - n_val

    excluded = set()
    for split, slice_ in zip(
        ("train", "val", "test"),
        (idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]),
    ):
        split_dir = root / split
        (split_dir / "GT").mkdir(parents=True, exist_ok=True)
        (split_dir / "NoisyLR").mkdir(parents=True, exist_ok=True)
        names = [gt_files[i] for i in sorted(slice_)]
        if split_dir.resolve() != src.resolve():
            for name in names:
                shutil.move(src / "GT" / name, split_dir / "GT" / name)
                shutil.move(src / "NoisyLR" / name, split_dir / "NoisyLR" / name)
        else:
            excluded.update(names)
        manifest[split] = {"count": len(names), "samples": names}
        print(f"{split}: {len(names)} samples")

    (root / "split.json").write_text(json.dumps(manifest, indent=2))

    for sub in ("GT", "NoisyLR"):
        leftover = [p.name for p in (src / sub).iterdir() if p.name not in excluded]
        for name in leftover:
            (src / sub / name).unlink()
        if not any((src / sub).iterdir()):
            os.rmdir(src / sub)


if __name__ == "__main__":
    main()