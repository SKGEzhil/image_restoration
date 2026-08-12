"""Cluster validation protocol (implementation.md §6 / design doc §7).

Loads a trained checkpoint and, on a held-out synthetic set, extracts the
deepest-stage cluster posterior (alpha / x_hat), runs t-SNE, colors by true
degradation order, and prints a cluster-purity table.

This is a diagnostic report only — nothing here feeds back into the model's
forward path, and the argmax used for the purity table is FOR ANALYSIS ONLY
(never for routing).

Run:
    python -m cgnafnet.validate_clusters \
        --config cgnafnet/configs/base.yaml \
        --checkpoint runs/<run_id>/best.pt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from cgnafnet.data.compose import ORDERS, sample_severity_and_order
from cgnafnet.data.dataset import CleanImageDataset, DegradedPairDataset
from cgnafnet.models.cg_nafnet import CGNAFNet
from cgnafnet.train import ROOT, load_config

ORDER_LABELS = ["->".join(o) for o in ORDERS]


def collect_posteriors(model, loader, device):
    """Run inference returning (deepest_alphas, order_labels, names)."""
    deepest, orders, names = [], [], []
    model.eval()
    with torch.no_grad():
        for degraded, _, logs, batch_names in tqdm(loader, desc="collect"):
            degraded = degraded.to(device)
            _, deep_alphas = model(degraded, return_cluster_posteriors=True)
            # deepest-stage alpha = final decoder stage posterior
            deepest.append(deep_alphas[-1].cpu())
            for log, name in zip(logs, batch_names):
                orders.append(tuple(log["order"]))
                names.append(name)
    return (
        torch.cat(deepest, 0).numpy(),
        np.array([ORDER_LABELS.index("->".join(o)) for o in orders]),
        names,
    )


def run_tsne(x, n_components=2, perplexity=30, seed=42):
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=seed,
                init="random", learning_rate="auto").fit_transform(x)


def purity_table(alpha, order_idx):
    """For each true order, dominant (argmax) cluster assignment distribution."""
    dominant = alpha.argmax(axis=1)
    print("\n=== cluster-purity table (argmax, analysis only) ===")
    print(f"{'order':<22} {'n':>5}  " + " ".join(f"c{i:>2}" for i in range(alpha.shape[1])))
    for oi, label in enumerate(ORDER_LABELS):
        mask = order_idx == oi
        counts = np.bincount(dominant[mask], minlength=alpha.shape[1])
        dist = " ".join(f"{c:>3}" for c in counts)
        purity = counts.max() / max(1, mask.sum())
        print(f"{label:<22} {mask.sum():>5}  {dist}  purity={purity:.2f}")


def plot_tsne(coords, order_idx, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for oi, label in enumerate(ORDER_LABELS):
        mask = order_idx == oi
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, label=label)
    ax.set_title("t-SNE of deepest-stage cluster posterior (by true order)")
    ax.legend(markerscale=2, fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved t-SNE plot: {out_path}")


def main(config_path, checkpoint_path, out_dir=None, num_samples=512):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if out_dir is None:
        out_dir = Path(checkpoint_path).resolve().parent / "cluster_validation"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg = cfg["model"]
    model = CGNAFNet(
        img_channel=1,
        width=mcfg["width"],
        num_stages=mcfg["num_stages"],
        blocks_per_stage=tuple(mcfg["blocks_per_stage"]),
        num_prototypes_per_stage=tuple(mcfg["num_prototypes_per_stage"]),
        prompt_dim=mcfg["prompt_dim"],
        proj_dim=mcfg.get("proj_dim", mcfg["prompt_dim"]),
        # Match training hparams so the state_dict loads; the head is never
        # called during diagnostics (true no-op by design).
        aux_order_head=mcfg.get("aux_order_head", False),
    ).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd["model"], strict=True)
    print(f"loaded checkpoint epoch {sd.get('epoch')}: {checkpoint_path}")

    dcfg = cfg["data"]
    clean_dirs = dcfg["clean_image_dirs"][:1]  # held-out split: use one dir
    ds = DegradedPairDataset(
        CleanImageDataset(clean_dirs),
        patch_size=dcfg["patch_size"],
        degradation_ranges=dcfg["degradation_ranges"],
        regional_mixing=False,
    )
    indices = torch.randperm(len(ds))[:num_samples].tolist()
    sub = torch.utils.data.Subset(ds, indices)
    loader = torch.utils.data.DataLoader(sub, batch_size=16, collate_fn=collate_items)

    alphas, order_idx, names = collect_posteriors(model, loader, device)
    print(f"collected {len(order_idx)} posteriors; order distribution: "
          f"{np.bincount(order_idx, minlength=6)}")

    purity_table(alphas, order_idx)
    coords = run_tsne(alphas)
    plot_tsne(coords, order_idx, out_dir / "tsne_by_order.png")


def collate_items(batch):
    degraded = torch.stack([b[0] for b in batch], 0)
    clean = torch.stack([b[1] for b in batch], 0)
    logs = [b[2] for b in batch]
    names = [b[3] for b in batch]
    return degraded, clean, logs, names


def parse_args():
    p = argparse.ArgumentParser(description="Validate CG-NAFNet clusters")
    p.add_argument("--config", default=str(ROOT / "cgnafnet" / "configs" / "base.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--num-samples", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.checkpoint, args.out_dir, args.num_samples)