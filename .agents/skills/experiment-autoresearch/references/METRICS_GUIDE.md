# Metrics Guide

How to interpret each metric. The agent uses this to make decisions.

## Available Metrics (from extract_metrics.py)

### Checkpoint-derived

| Key | Meaning | Good = |
|-----|---------|--------|
| `best_val_loss` | Lowest validation loss achieved | lower |
| `best_epoch` | Epoch where best loss occurred | earlier = faster convergence |
| `best_step` / `best_global_step` | Step count at best epoch | |

### Final values (from last log point)

| Key | Meaning | Good = |
|-----|---------|--------|
| `final_val_ssim` | Last validation SSIM | higher (max 1.0) |
| `final_val_psnr` | Last validation PSNR | higher (dB) |
| `final_val_loss` | Last validation loss | lower |
| `final_val_l1` | Last validation L1 | lower |
| `final_train_ssim` | Last training SSIM | for train/val gap |
| `final_train_loss` | Last training loss | for train/val gap |

### Series (for trend analysis)

| Key | Format | Use |
|-----|--------|-----|
| `val_ssim_series` | `[[step, value], ...]` | Feed to basic_stats.py for slope |
| `val_psnr_series` | `[[step, value], ...]` | |
| `val_loss_series` | `[[step, value], ...]` | |
| `train_ssim_series` | `[[step, value], ...]` | For overfitting detection |
| `train_loss_series` | `[[step, value], ...]` | For overfitting detection |

### Derived

| Key | Meaning | Typical ranges |
|-----|---------|---------------|
| `num_val_points` | How many validation snapshots | More = better resolution |
| `time_per_epoch_sec` | Wall-clock per epoch | For efficiency comparison |
| `last_log_lines` | Last 10 log lines | For crash/error diagnosis |

## How to Detect Issues

### Good Run
- `final_val_ssim` rising over time (check series slope > 0)
- `final_val_loss` decreasing
- Train/val gap = `final_train_ssim - final_val_ssim` < 0.02
- `instability.spike_count` from basic_stats.py < 2

### Overfitting
- `final_train_ssim` >> `final_val_ssim` (gap > 0.03)
- `val_loss_series` starts rising while `train_loss_series` still falls
- Action: add regularization, reduce model size, or stop early

### Underfitting / Too Slow
- Both train and val metrics flat (slope ≈ 0 from basic_stats.py)
- No improvement after many epochs
- Action: increase LR, increase capacity, or change loss

### Unstable / Bad Config
- `val_loss_series` spikes up suddenly
- `train_loss_series` oscillates wildly
- Action: lower LR, reduce batch size, add gradient clipping

### Still Rising at Budget Cutoff
- basic_stats.py says `trend = "improving"` or `"strongly_improving"`
- basic_stats.py says `convergence.converged = false`
- Action: extend epochs — do NOT kill. SCUNet has late crossover.

## Decision Patterns

1. **Best final val_ssim** — simple ranking by highest SSIM
2. **Best with positive slope** — prefers runs still improving (use basic_stats.py slope)
3. **Stable winner** — penalize large train/val gap and spikes
4. **Efficiency** — val_ssim / time_per_epoch for best bang/buck
5. **Conservative** — among top 2, pick simpler/lower config

## Test Metrics (from run_test.py)

| Key | Meaning |
|-----|---------|
| `test_metrics.SSIM` | Test set SSIM |
| `test_metrics.PSNR` | Test set PSNR |
| `test_metrics.L1` | Test set L1 |
| `test_metrics.LPIPS` | Test set LPIPS |

Test metrics are the ground truth — val metrics guide selection, test metrics
confirm generalization.
