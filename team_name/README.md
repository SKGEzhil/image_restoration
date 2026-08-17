# Image Restoration — KLA Hackathon 2026 Submission

SCUNetSR-based image restoration for semiconductor inspection. Restores degraded noisy, low-resolution images to clean high-resolution outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python run.py <input_dir> <output_dir>
```

## Usage

```bash
python run.py test/NoisyLR results/
```

**Arguments:**
- `input_dir` — Directory containing degraded `.npy` images (128×128, float32)
- `output_dir` — Directory to write restored `.npy` images (256×256, float32)

## Input/Output Format

| Property | Input | Output |
|----------|-------|--------|
| Format | `.npy` (NumPy) | `.npy` (NumPy) |
| Resolution | 128×128 | 256×256 |
| Channels | 1 (grayscale) | 1 (grayscale) |
| Dtype | float32 | float32 |
| Value range | May exceed [0,1] | Clamped to [0,1] |

## Model

- **Architecture:** SCUNetSR (Swin-Conv-UNet + PixelShuffle)
- **Type:** Hybrid CNN + Swin Transformer with 2× super-resolution
- **Parameters:** ~17M
- **Test-Time Augmentation (TTA):** Enabled by default (8-augmentation average: 4 rotations × 2 flips)

## Directory Structure

```
team_name/
├── run.py              # Inference script
├── requirements.txt    # Dependencies
├── README.md           # This file
├── models/
│   ├── __init__.py
│   └── scunet.py       # SCUNetSR model definition
└── weights/
    └── best.pt         # Model checkpoint
```
