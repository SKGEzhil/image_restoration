"""Render GT/NoisyLR/Output grids grouped by LPIPS buckets."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LPIPS_BUCKETS = [
    ("<0.1", lambda v: v < 0.1),
    ("0.1-0.2", lambda v: (v >= 0.1) & (v < 0.2)),
    ("0.2-0.3", lambda v: (v >= 0.2) & (v < 0.3)),
    ("0.3-0.5", lambda v: (v >= 0.3) & (v < 0.5)),
    (">0.5", lambda v: v >= 0.5),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create LPIPS-bucketed comparison grids from a test run"
    )
    parser.add_argument(
        "run_path",
        help="Path to runs/<run_id> or directly to runs/<run_id>/test_details.json",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset root containing test/GT and test/NoisyLR",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples per PNG",
    )
    return parser.parse_args()


def resolve_run_dir(run_path: str) -> Path:
    p = Path(run_path)
    if p.is_file():
        if p.name != "test_details.json":
            raise ValueError(f"Unsupported file: {p}")
        return p.parent
    return p


def resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        p = Path(data_dir)
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset root does not exist: {p}")

    candidates = [
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent.parent / "data",
    ]
    for candidate in candidates:
        if (candidate / "test" / "GT").exists() and (candidate / "test" / "NoisyLR").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate dataset root. Pass --data-dir pointing to a folder containing test/GT and test/NoisyLR."
    )


def load_details(run_dir: Path):
    details_file = run_dir / "test_details.json"
    if not details_file.exists():
        raise FileNotFoundError(f"Missing test_details.json: {details_file}")
    details = json.loads(details_file.read_text())
    per_sample = details.get("per_sample", [])
    if not per_sample:
        raise ValueError(f"No per_sample entries found in {details_file}")
    return details, per_sample


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path).squeeze()


def bucket_samples(per_sample):
    groups = {label: [] for label, _ in LPIPS_BUCKETS}
    for entry in per_sample:
        lpips = float(entry["LPIPS"])
        for label, predicate in LPIPS_BUCKETS:
            if predicate(lpips):
                groups[label].append(entry)
                break
    for label in groups:
        groups[label].sort(key=lambda row: float(row["LPIPS"]))
    return groups


def make_output_dirs(run_dir: Path):
    root = run_dir / "lpips_visualizations"
    root.mkdir(parents=True, exist_ok=True)
    bucket_dirs = {}
    for label, _ in LPIPS_BUCKETS:
        bucket_dir = root / label
        bucket_dir.mkdir(parents=True, exist_ok=True)
        bucket_dirs[label] = bucket_dir
    return root, bucket_dirs


def render_batch(batch, data_dir: Path, outputs_dir: Path, out_file: Path, title: str):
    n = len(batch)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 9))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for col, entry in enumerate(batch):
        name = entry["name"]
        stem = Path(name).stem
        gt = load_image(data_dir / "test" / "GT" / name)
        noisy = load_image(data_dir / "test" / "NoisyLR" / name)
        output = load_image(outputs_dir / name)
        lpips = float(entry["LPIPS"])
        ssim = float(entry["SSIM"])
        psnr = float(entry["PSNR"])

        images = [gt, noisy, output]
        row_labels = ["GT", "NoisyLR", "Output"]
        for row, (img, row_label) in enumerate(zip(images, row_labels)):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{stem}\nLPIPS {lpips:.4f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, rotation=0, labelpad=28, va="center")

        axes[2, col].text(
            0.5,
            -0.18,
            f"SSIM {ssim:.4f}\nPSNR {psnr:.2f} dB",
            transform=axes[2, col].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_path)
    details, per_sample = load_details(run_dir)
    data_dir = resolve_data_dir(args.data_dir)
    outputs_dir = run_dir / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Missing outputs directory: {outputs_dir}")

    _, bucket_dirs = make_output_dirs(run_dir)
    grouped = bucket_samples(per_sample)

    for label, samples in grouped.items():
        bucket_dir = bucket_dirs[label]
        if not samples:
            continue
        for batch_index, start in enumerate(range(0, len(samples), args.batch_size), start=1):
            batch = samples[start:start + args.batch_size]
            out_file = bucket_dir / f"batch_{batch_index:03d}.png"
            title = (
                f"{run_dir.name} | LPIPS {label} | "
                f"samples {start + 1}-{start + len(batch)} of {len(samples)}"
            )
            render_batch(batch, data_dir, outputs_dir, out_file, title)
            print(out_file)

    print(f"saved under: {run_dir / 'lpips_visualizations'}")
    print(f"checkpoint: {details.get('checkpoint', 'unknown')}")


if __name__ == "__main__":
    main()
