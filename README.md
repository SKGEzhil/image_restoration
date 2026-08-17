# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**KLA Hackathon 2026 — SEMICON India**

A deep learning pipeline that restores degraded noisy, low-resolution images to clean high-resolution outputs. Handles speckle noise, additive Gaussian noise, and downsampling degradations.

## Repository Structure

```
├── standalone_inference.py   # Self-contained inference script (no src/ dependency)
├── requirements.txt          # Python dependencies (CUDA 12.8)
├── src/
│   ├── train.py              # Training script
│   ├── test.py               # Evaluation / submission script
│   ├── dataset.py            # Paired dataset loader
│   ├── losses.py             # Configurable loss system
│   ├── metrics.py            # PSNR, SSIM, DISTS evaluation metrics
│   ├── models/
│   │   ├── scunet.py         # SCUNet / SCUNetSR architecture
│   │   ├── nafnet.py         # NAFNet / NAFNetSR architecture
│   │   └── discriminator.py  # PatchGAN discriminator (GAN training)
│   ├── training_config.yaml  # Training hyperparameters
│   ├── test_config.yaml      # Test/evaluation config
│   └── loss_presets.json     # Loss preset definitions
├── configs/                  # Additional config files
├── weights/                  # Model checkpoints (add your .pt files here)
├── results/                  # Sample restored outputs
└── data/                     # Dataset (not in repo, see Dataset section)
    ├── train/
    │   ├── NoisyLR/          # 128x128 degraded .npy files
    │   └── GT/               # 256x256 ground truth .npy files
    ├── val/
    │   ├── NoisyLR/
    │   └── GT/
    └── test/
        └── NoisyLR/          # Test inputs (no GT)
```

## Environment Setup

**Requirements:** Python ≥ 3.13, NVIDIA GPU with CUDA 12.8

```bash
# Clone the repository
git clone <repo-url>
cd Image_Restoration

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies (includes PyTorch with CUDA 12.8)
pip install -r requirements.txt
```

Verify CUDA is available:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Dataset

Download the official KLA dataset and place it under `data/` with this structure:

```
data/
├── train/
│   ├── NoisyLR/       # 128x128 float32 .npy (degraded, values may exceed [0,1])
│   └── GT/            # 256x256 float32 .npy (clean, values in [0,1])
├── val/
│   ├── NoisyLR/
│   └── GT/
└── test/
    └── NoisyLR/       # Test inputs only (no GT provided)
```

- **GT:** normalized to [0, 1]
- **NoisyLR:** may extend slightly outside [0, 1] — this is intentional, do not clip inputs
- **Degradations:** speckle noise, additive Gaussian noise, downsampling (any order)
- **Image sizes:** ~256x256 or 512x512 at evaluation; training pairs are 128→256

## Training

```bash
# Basic training with default config (SCUNetSR, L1 loss)
python src/train.py

# Override config via CLI
python src/train.py --loss-preset scunet_l1 --epochs 40 --batch-size 12 --lr 0.0001

# Use a custom config file
python src/train.py --config path/to/custom_config.yaml

# Resume from checkpoint
python src/train.py --resume runs/<run_id>/last.pt

# Named run
python src/train.py --run-name my_experiment_01
```

Checkpoints are saved to `runs/<run_id>/`:
- `best.pt` — best validation loss checkpoint
- `last.pt` — latest epoch checkpoint
- `config.json` — full training args snapshot

### Model Architectures

| Model | Key | Description |
|-------|-----|-------------|
| SCUNetSR | `scunet_sr` | Swin-Conv-UNet with PixelShuffle head (128→256). Default. |
| SCUNetSR-real | `scunet_sr_real` | Heavier SCUNetSR variant (`config=[4,4,4,4,4,4,4]`) |
| NAFNetSR | `nafnet` | Nonlinear Activation-Free Network with SR head |

Select model in config or override:
```yaml
# In training_config.yaml
train_model: scunet_sr
models:
  scunet_sr:
    in_nc: 1
    config: [2, 2, 2, 2, 2, 2, 2]
    dim: 64
    drop_path_rate: 0.0
    input_resolution: 128
    up_scale: 2
```

### Loss Presets

Presets are defined in `src/loss_presets.json`. Key options:

| Preset | Components | Use Case |
|--------|-----------|----------|
| `scunet_l1` | L1 only | Fast baseline, SCUNet paper default |
| `l1_ssim_baseline` | L1 + SSIM | Balanced pixel + structural loss |
| `combo_1` | L1 + SSIM + FFT | Adds frequency-domain supervision |
| `combo_3` | Charbonnier + MS-SSIM + Gradient + FFL | Multi-component, uncertainty-weighted |

```bash
python src/train.py --loss-preset combo_3
```

### GAN Training (Optional)

Enable adversarial training in `training_config.yaml`:
```yaml
gan_training:
  enabled: true
  pretrain_epochs: 10       # L1-only warmup
  adv_warmup_epochs: 5      # ramp adversarial weight
  discriminator_update_freq: 2
  feature_matching: true
```

## Evaluation / Testing

```bash
# Evaluate on test split with GT (paired metrics)
python src/test.py --checkpoint runs/<run_id>/best.pt

# With test-time augmentation (8-aug average, higher quality)
python src/test.py --checkpoint runs/<run_id>/best.pt --use-tta

# Build submission CSV (no GT required)
python src/test.py --checkpoint runs/<run_id>/best.pt --build-submission --use-tta

# Save restored .npy outputs
python src/test.py --checkpoint runs/<run_id>/best.pt --save-outputs
```

**Reported metrics:** PSNR (dB), SSIM, DISTS, L1

## Standalone Inference

A self-contained inference script that requires **no imports from `src/`** and **no manual setup**. Evaluators can run it directly.

```bash
# Basic usage
python standalone_inference.py --input-dir test/NoisyLR --output-dir results/

# With custom checkpoint and batch size
python standalone_inference.py \
    --input-dir /path/to/degraded/images \
    --output-dir /path/to/save/restored \
    --checkpoint weights/best.pt \
    --batch-size 8

# Force device
python standalone_inference.py --input-dir test/NoisyLR --output-dir results/ --device cuda
```

**Arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--input-dir` | required | Directory containing `.npy` NoisyLR images |
| `--output-dir` | required | Directory to write restored `.npy` images |
| `--checkpoint` | `weights/best.pt` | Path to model checkpoint |
| `--batch-size` | `16` | Inference batch size |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

The script automatically:
- Loads the SCUNetSR architecture from the checkpoint metadata
- Strips `torch.compile` key prefixes (`_orig_mod.`) if present
- Runs test-time augmentation (8-aug: 4 rotations × 2 flips) for best quality
- Prints per-image and total inference time

## Experiment Tracking

Weights & Biases integration is built in. To enable:

```bash
# Set your API key in .env
echo "WANDB_API_KEY=your_key_here" > .env
```

Metrics logged: train/loss, train/ssim, train/lr, val/loss, val/psnr, val/ssim, sample images.

## Hardware & Software

- **GPU:** NVIDIA GPU with CUDA 12.8 (tested on H100 for evaluation)
- **Python:** ≥ 3.13
- **PyTorch:** ≥ 2.13.0
- **OS:** Linux (primary), Windows, macOS (CPU/MPS fallback)

## License

This project is developed for the KLA Hackathon 2026 competition. See individual dependency licenses for third-party components:
- [SCUNet](https://github.com/cszn/SCUNet) — Apache 2.0
- [NAFNet](https://github.com/megvii-research/NAFNet) — Apache 2.0
- [kornia](https://github.com/kornia/kornia) — Apache 2.0
- [timm](https://github.com/huggingface/pytorch-image-models) — Apache 2.0
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) — BSD-2
