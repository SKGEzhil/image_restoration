"""Evaluate a trained NAFNet on data/test.

Computes the same metrics used during training (L1 + SSIM) plus PSNR.
Usage:
    python test.py --checkpoint runs/<run_id>/best.pt
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedDataset, get_device
from metrics import compute_psnr, separate_losses
from model import NAFNetSR


def parse_args():
    p = argparse.ArgumentParser(description="Test NAFNet on the test split")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained checkpoint .pt file")
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-outputs", action="store_true",
                   help="Save restored samples as .npy into runs/<run_id>/outputs")
    return p.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    model = NAFNetSR(
        up_scale=2,
        width=cfg.get("width", 32),
        num_blks=cfg.get("num_blks", 8),
        img_channel=1,
        drop_out_rate=cfg.get("drop_out_rate", 0.0),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


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

    model, ckpt = load_model(args.checkpoint, device)
    logger.info(f"model loaded (train global_step={ckpt.get('global_step')}, "
                f"best_val_loss={ckpt.get('best_val_loss')})")

    test_ds = PairedDataset(args.data_dir, split="test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type != "mps",
    )

    model.eval()
    total_l1 = total_ssim_loss = total_psnr = 0.0
    num = 0
    outputs = []
    start = time.time()
    with torch.no_grad():
        for lr, gt, names in tqdm(test_loader, desc="Testing", unit="batch"):
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr)
            l1, ssim_loss, _ = separate_losses(pred, gt)
            b = lr.size(0)
            total_l1 += l1.item() * b
            total_ssim_loss += ssim_loss.item() * b
            total_psnr += compute_psnr(pred, gt).item() * b
            num += b
            if args.save_outputs:
                for name, out in zip(names, pred.clamp(0, 1).cpu().numpy()):
                    outputs.append((name, out))

    metrics = {
        "checkpoint": args.checkpoint,
        "num_samples": num,
        "L1": total_l1 / num,
        "SSIM": 1.0 - total_ssim_loss / num,
        "SSIM_loss": total_ssim_loss / num,
        "PSNR": total_psnr / num,
        "elapsed_s": round(time.time() - start, 2),
    }

    out_cfg = ckpt.get("args", {})
    metrics_file = out_dir / "test_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info(metrics_file)

    print("\n=== Test results ===")
    print(f"  samples : {metrics['num_samples']}")
    print(f"  L1      : {metrics['L1']:.4f}")
    print(f"  SSIM    : {metrics['SSIM']:.4f}")
    print(f"  PSNR    : {metrics['PSNR']:.2f} dB")
    print(f"  saved   : {metrics_file}")

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