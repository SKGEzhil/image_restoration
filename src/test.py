"""Evaluate a trained model on data/test.

Computes the same metrics used during training (L1 + SSIM) plus PSNR and LPIPS.
Usage:
    python test.py --checkpoint runs/<run_id>/best.pt
    python test.py --checkpoint runs/<run_id>/best.pt --batch-size 32
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedDataset, get_device
from metrics import compute_lpips, compute_psnr, separate_losses
from models import create_model


DEFAULT_CONFIG = Path(__file__).resolve().parent / "test_config.yaml"


def parse_args():
    # Load config as defaults
    with open(DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser(description="Test model on the test split")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained checkpoint .pt file")
    p.add_argument("--test-model", type=str, default=config.get("test_model", "nafnet"),
                   help="Model name (overrides test_config.yaml)")
    p.add_argument("--data-dir", type=str, default=config.get("data_dir", "data"),
                   help="Data directory (overrides test_config.yaml)")
    p.add_argument("--batch-size", type=int, default=config.get("batch_size", 16),
                   help="Batch size (overrides test_config.yaml)")
    p.add_argument("--num-workers", type=int, default=config.get("num_workers", 2),
                   help="DataLoader workers (overrides test_config.yaml)")
    p.add_argument("--save-outputs", action="store_true", default=config.get("save_outputs", False),
                   help="Save restored samples as .npy into runs/<run_id>/outputs")
    args = p.parse_args()

    # Store full config for model params lookup
    args.config = config
    return args


def load_model(checkpoint_path, args, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})

    # Model name: prefer checkpoint's saved train_model, fallback to CLI arg
    model_name = cfg.get("train_model", args.test_model)
    # Model params: prefer checkpoint's saved models dict, fallback to test_config
    model_params = cfg.get("models", {}).get(model_name,
                       args.config.get("models", {}).get(model_name, {}))

    model = create_model(name=model_name, **model_params)
    state_dict = ckpt["model"]
    # torch.compile prefixes keys with _orig_mod.; strip for vanilla model
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    return model.to(device), ckpt, model_name


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(f"logs/log_{out_dir.name}_test.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger("test")
    logger.info(f"checkpoint={args.checkpoint}")
    logger.info(f"device={device}")

    model, ckpt, model_name = load_model(args.checkpoint, args, device)
    logger.info(f"model={model_name} loaded (train global_step={ckpt.get('global_step')}, "
                f"best_val_loss={ckpt.get('best_val_loss')})")

    test_ds = PairedDataset(args.data_dir, split="test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type != "mps",
    )

    model.eval()
    total_l1 = total_ssim_loss = total_psnr = total_lpips = 0.0
    num = 0
    outputs = []
    per_sample = []
    start = time.time()
    with torch.no_grad():
        for lr, gt, names in tqdm(test_loader, desc="Testing", unit="batch"):
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr)
            l1, ssim_loss, _, freq_loss = separate_losses(pred, gt)
            b = lr.size(0)
            total_l1 += l1.item() * b
            total_ssim_loss += ssim_loss.item() * b
            total_psnr += compute_psnr(pred, gt).item() * b
            total_lpips += compute_lpips(pred, gt).item() * b
            d = compute_lpips(gt, gt)  # identical image vs itself
            print("D:", d)  # must be ~0.0000
            num += b
            if args.save_outputs:
                for name, out in zip(names, pred.clamp(0, 1).cpu().numpy()):
                    outputs.append((name, out))
            for name, p, g in zip(names, pred, gt):
                l1_s, ssim_loss_s, ssim_s, _ = separate_losses(p.unsqueeze(0), g.unsqueeze(0))
                psnr_s = compute_psnr(p.unsqueeze(0), g.unsqueeze(0))
                lpips_s = compute_lpips(p.unsqueeze(0), g.unsqueeze(0))
                per_sample.append({
                    "name": name,
                    "L1": round(float(l1_s), 6),
                    "SSIM": round(float(ssim_s), 6),
                    "SSIM_loss": round(float(ssim_loss_s), 6),
                    "PSNR": round(float(psnr_s), 6),
                    "LPIPS": round(float(lpips_s), 6),
                })

    metrics = {
        "checkpoint": args.checkpoint,
        "num_samples": num,
        "L1": total_l1 / num,
        "SSIM": 1.0 - total_ssim_loss / num,
        "SSIM_loss": total_ssim_loss / num,
        "PSNR": total_psnr / num,
        "LPIPS": total_lpips / num,
        "elapsed_s": round(time.time() - start, 2),
    }

    out_cfg = ckpt.get("args", {})
    metrics_file = out_dir / "test_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info(metrics_file)

    details_file = out_dir / "test_details.json"
    details = {
        "checkpoint": args.checkpoint,
        "num_samples": num,
        "metrics": metrics,
        "per_sample": per_sample,
    }
    details_file.write_text(json.dumps(details, indent=2))
    logger.info(details_file)

    print("\n=== Test results ===")
    print(f"  samples : {metrics['num_samples']}")
    print(f"  L1      : {metrics['L1']:.4f}")
    print(f"  SSIM    : {metrics['SSIM']:.4f}")
    print(f"  PSNR    : {metrics['PSNR']:.2f} dB")
    print(f"  LPIPS   : {metrics['LPIPS']:.4f}")
    print(f"  saved   : {metrics_file}")
    print(f"  details : {details_file}")

    if args.save_outputs:
        out_dir_outputs = out_dir / "outputs"
        out_dir_outputs.mkdir(parents=True, exist_ok=True)
        for name, out in outputs:
            np.save(out_dir_outputs / name, out)
        print(f"  outputs : {out_dir_outputs}")

    load_dotenv()
    key = os.getenv("WANDB_API_KEY")
    if key and key != "YOUR_WANDB_API_KEY_HERE":
        import wandb
        os.environ["WANDB_API_KEY"] = key
        ckpt_run_id = Path(args.checkpoint).parent.name
        run = wandb.init(
            project="image-restoration",
            id=f"{ckpt_run_id}_test",
            name=f"{ckpt_run_id}/test",
            config=out_cfg,
            notes=f"Evaluation on test split from checkpoint {args.checkpoint}",
        )
        run.log(metrics)
        run.finish()


if __name__ == "__main__":
    main()