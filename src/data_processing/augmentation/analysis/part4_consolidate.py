"""Part 4: Consolidate Findings — merge all part results into a final report.

Generates a single JSON and Markdown report summarizing all estimated
degradation properties.
"""

import json
from pathlib import Path

import numpy as np

from utils import ensure_output_dir, load_config


def load_part_summary(output_dir: Path, part_name: str) -> dict:
    """Load a part's summary JSON."""
    path = output_dir / f"{part_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"{part_name} summary not found at {path}")
    with open(path) as f:
        return json.load(f)


def consolidate(config: dict) -> dict:
    """Load all part summaries and produce a consolidated report."""
    output_dir = ensure_output_dir(config)

    part1 = load_part_summary(output_dir, "part1")
    part2 = load_part_summary(output_dir, "part2")
    part3 = load_part_summary(output_dir, "part3")

    # Build consolidated report
    report = {
        "dataset": {
            "split": config["split"],
            "total_images": part1["total_images"],
            "downsample_factor": config["downsample_factor"],
        },
        "downsampling": {
            "winner": part1["overall_winner"],
            "kernel_win_rates": part1["kernel_win_rates"],
        },
        "noise": {
            "gaussian_sigma_range_p5_p95": part2["gaussian_noise"]["sigma_range_p5_p95"],
            "gaussian_sigma_distribution": part2["gaussian_noise"]["sigma_distribution"],
            "speckle_sigma_range_p5_p95": part2["speckle_noise"]["sigma_range_p5_p95"],
            "speckle_sigma_distribution": part2["speckle_noise"]["sigma_distribution"],
            "cluster_proportions": part2["cluster_proportions"],
            "cluster_centers": part2["cluster_centers"],
        },
        "order": {
            "spectral_proportions": part3["spectral_proportions"],
            "threshold_used": part3["threshold_used"],
        },
        "value_range": {
            "gt": [0.0, 1.0],
            "noisy_lr": "unclamped (may exceed [0,1])",
        },
    }

    # Add gaussian sigma range if Part 1 found gaussian-decimate wins
    if part1.get("gaussian_sigma_distribution"):
        report["downsampling"]["gaussian_sigma_range"] = [
            part1["gaussian_sigma_distribution"]["p5"],
            part1["gaussian_sigma_distribution"]["p95"],
        ]
        report["downsampling"]["gaussian_sigma_stats"] = part1["gaussian_sigma_distribution"]

    # Save JSON
    json_path = output_dir / "analysis_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Generate Markdown report
    md_lines = [
        "# Degradation Estimation Report",
        "",
        "## Dataset",
        f"- Split: `{report['dataset']['split']}`",
        f"- Total images: {report['dataset']['total_images']}",
        f"- Downsample factor: {report['dataset']['downsample_factor']}x",
        f"- GT value range: {report['value_range']['gt']}",
        f"- NoisyLR value range: {report['value_range']['noisy_lr']}",
        "",
        "## Downsampling Kernel",
        f"- **Winner: {report['downsampling']['winner']}**",
        "",
        "| Kernel | Count | Percentage |",
        "|--------|-------|------------|",
    ]

    for kname, kinfo in report["downsampling"]["kernel_win_rates"].items():
        md_lines.append(f"| {kname} | {kinfo['count']} | {kinfo['pct']:.1f}% |")

    if "gaussian_sigma_stats" in report["downsampling"]:
        gs = report["downsampling"]["gaussian_sigma_stats"]
        md_lines.extend([
            "",
            "### Gaussian Blur Sigma Distribution",
            f"- Count: {gs['count']}",
            f"- Range (p5-p95): [{gs['p5']:.3f}, {gs['p95']:.3f}]",
            f"- Mean: {gs['mean']:.3f}, Std: {gs['std']:.3f}",
        ])

    md_lines.extend([
        "",
        "## Noise Levels",
        "",
        "### Gaussian (Additive) Noise",
        f"- Sigma range (p5-p95): [{report['noise']['gaussian_sigma_range_p5_p95'][0]:.4f}, "
        f"{report['noise']['gaussian_sigma_range_p5_p95'][1]:.4f}]",
        f"- Mean: {report['noise']['gaussian_sigma_distribution']['mean']:.4f}",
        "",
        "### Speckle (Multiplicative) Noise",
        f"- Sigma range (p5-p95): [{report['noise']['speckle_sigma_range_p5_p95'][0]:.4f}, "
        f"{report['noise']['speckle_sigma_range_p5_p95'][1]:.4f}]",
        f"- Mean: {report['noise']['speckle_sigma_distribution']['mean']:.4f}",
        "",
        "### Cluster Proportions",
        "",
        "| Cluster | Count | Percentage |",
        "|---------|-------|------------|",
    ])

    for ct, info in report["noise"]["cluster_proportions"].items():
        md_lines.append(f"| {ct} | {info['count']} | {info['pct']:.1f}% |")

    md_lines.extend([
        "",
        "## Degradation Order (Spectral Analysis)",
        "",
        "| Signature | Count | Percentage |",
        "|-----------|-------|------------|",
    ])

    for label, info in report["order"]["spectral_proportions"].items():
        md_lines.append(f"| {label} | {info['count']} | {info['pct']:.1f}% |")

    md_lines.extend([
        "",
        f"Threshold used: {report['order']['threshold_used']}",
        "",
    ])

    md_path = output_dir / "analysis_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return report, md_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 4: Consolidate findings")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    report, md_path = consolidate(config)

    print("\n=== Part 4: Consolidated Report ===")
    print(f"JSON: analysis_report.json")
    print(f"Markdown: {md_path}")
    print(f"\nOverall winner: {report['downsampling']['winner']}")
    print(f"Gaussian sigma (p5-p95): {report['noise']['gaussian_sigma_range_p5_p95']}")
    print(f"Speckle sigma (p5-p95): {report['noise']['speckle_sigma_range_p5_p95']}")
    for label, info in report["order"]["spectral_proportions"].items():
        print(f"  {label}: {info['pct']:.1f}%")
