"""Extract metrics from a completed training run.

Reads the checkpoint (best.pt) and log file to produce a structured JSON summary.
This is a dumb utility — the agent decides what to do with the results.
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def parse_val_line(line: str) -> dict | None:
    """Parse a validation log line like:
    [epoch 5/40 step 32/200 global_step 1032] val/loss=0.0123 val/psnr=26.45 val/ssim=0.7621
    """
    epoch_match = re.search(r"epoch\s+(\d+)", line)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    step_match = re.search(r"global_step\s+(\d+)", line)
    global_step = int(step_match.group(1)) if step_match else None

    metrics = {"epoch": epoch, "global_step": global_step}
    for m in re.finditer(r"(val|train)/(\w+)=([0-9.eE+-]+)", line):
        prefix, name, val = m.groups()
        metrics[f"{prefix}/{name}"] = float(val)

    return metrics if len(metrics) > 2 else None


def load_checkpoint_metrics(run_dir: Path) -> dict:
    """Load metrics from best.pt checkpoint."""
    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        return {}

    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)

    result = {
        "best_val_loss": float(ckpt.get("best_val_loss", 0.0)),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "best_step": int(ckpt.get("step", 0)),
        "best_global_step": int(ckpt.get("global_step", 0)),
    }

    val_metrics = ckpt.get("val_metrics", {})
    for k, v in val_metrics.items():
        result[k] = float(v)

    args = ckpt.get("args", {})
    result["config_snapshot"] = {k: v for k, v in args.items() if k not in ("run_name",)}

    return result


def parse_log_file(log_file: Path) -> tuple[list[dict], list[dict]]:
    """Parse both validation and train metrics from log file.
    Returns (val_points, train_points) where each point is a dict of metrics.
    """
    if not log_file.exists():
        return [], []

    val_points = []
    train_points = []

    with open(log_file) as f:
        for line in f:
            if "val/" in line:
                parsed = parse_val_line(line)
                if parsed and any(k.startswith("val/") for k in parsed):
                    val_points.append(parsed)
            if "train/" in line:
                parsed = parse_val_line(line)
                if parsed and any(k.startswith("train/") for k in parsed):
                    train_points.append(parsed)

    return val_points, train_points


def extract_time_per_epoch(log_file: Path) -> float | None:
    """Estimate wall-clock time per epoch from log timestamps."""
    if not log_file.exists():
        return None

    timestamps = []
    with open(log_file) as f:
        for line in f:
            m = re.match(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),?\d*", line)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    timestamps.append(ts)
                except ValueError:
                    pass

    if len(timestamps) < 2:
        return None

    timestamps.sort()
    total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()

    max_epoch = 0
    with open(log_file) as f:
        for line in f:
            m = re.search(r"epoch\s+(\d+)", line)
            if m:
                max_epoch = max(max_epoch, int(m.group(1)))

    if max_epoch > 0 and total_seconds > 0:
        return total_seconds / max_epoch

    return None


def get_last_lines(log_file: Path, n: int = 20) -> list[str]:
    """Read last N lines from log file."""
    if not log_file.exists():
        return []
    with open(log_file) as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n * 200))
        return f.readlines()[-n:]


def extract_all_metrics(run_name: str, repo_root: Path) -> dict:
    """Extract complete metrics for a run. Returns a flat dict the agent can read."""
    run_dir = repo_root / "runs" / run_name
    log_file = repo_root / "logs" / f"log_{run_name}.log"

    result = {"run_name": run_name}

    # 1. Checkpoint metrics
    ckpt_metrics = load_checkpoint_metrics(run_dir)
    result["checkpoint_found"] = bool(ckpt_metrics)
    if ckpt_metrics:
        result["best_val_loss"] = ckpt_metrics.get("best_val_loss")
        result["best_epoch"] = ckpt_metrics.get("best_epoch")
        result["best_step"] = ckpt_metrics.get("best_step")
        result["best_global_step"] = ckpt_metrics.get("best_global_step")
        # Copy all val/* metrics from checkpoint
        for k, v in ckpt_metrics.items():
            if k.startswith("val/") or k.startswith("train/"):
                result[k] = v
        result["config_snapshot"] = ckpt_metrics.get("config_snapshot", {})

    # 2. Log-based history
    val_points, train_points = parse_log_file(log_file)
    result["num_val_points"] = len(val_points)
    result["num_train_points"] = len(train_points)

    # 3. Extract metric series from val_points
    metric_names = set()
    for p in val_points:
        metric_names.update(k for k in p if k.startswith("val/") or k in ("epoch", "global_step"))
    for p in train_points:
        metric_names.update(k for k in p if k.startswith("train/") or k in ("epoch", "global_step"))

    # Build series dicts
    for prefix in ("val", "train"):
        for metric_name in metric_names:
            if not metric_name.startswith(f"{prefix}/"):
                continue
            series_key = f"{prefix}_{metric_name.split('/')[1]}_series"
            points = val_points if prefix == "val" else train_points
            series = [(p.get("global_step", p.get("epoch", i)), p.get(metric_name)) for i, p in enumerate(points) if metric_name in p]
            if series:
                result[series_key] = series

    # 4. Final metric values from log (last validation point)
    if val_points:
        last = val_points[-1]
        for k, v in last.items():
            if k.startswith("val/"):
                result[f"final_{k.replace('/', '_')}"] = v

    if train_points:
        last = train_points[-1]
        for k, v in last.items():
            if k.startswith("train/"):
                result[f"final_{k.replace('/', '_')}"] = v

    # 5. Time per epoch
    tpe = extract_time_per_epoch(log_file)
    if tpe is not None:
        result["time_per_epoch_sec"] = round(tpe, 1)

    # 6. Test metrics
    test_metrics_file = run_dir / "test_metrics.json"
    if test_metrics_file.exists():
        try:
            test_data = json.loads(test_metrics_file.read_text())
            result["test_metrics"] = test_data
        except json.JSONDecodeError:
            pass

    # 7. Last log lines for agent context (error detection)
    last_lines = get_last_lines(log_file, 10)
    if last_lines:
        result["last_log_lines"] = [line.strip() for line in last_lines if line.strip()]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=None, help="Write JSON to file instead of stdout")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metrics = extract_all_metrics(args.run_name, repo_root)

    out = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
