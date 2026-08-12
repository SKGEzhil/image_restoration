# CG-NAFNet

Cluster-Guided Dynamic NAFNet — a single-pass, real-time image restoration
model for images corrupted by an **unknown-order** composition of speckle
noise, additive Gaussian noise, and downsampling.

Implementation spec: `implementation.md` (repo root). Architectural rationale:
`clusir_nafnet_restoration_design.md` (external).

## Why it exists

Classic single-task restoration assumes one known degradation. Real DI
(digital imaging) data rarely obeys that; instead speckle, sensor noise and
resolution loss arrive in an unknown order and mixture per image. NG-RAFNet
solves this with a standard NAFNet backbone plus per-stage **soft cluster
routing** (PCGRMLite posterior → reparameterized degradation prompt → zero-init
FiLM), so the *same* weights flexibly restore different degradation
compositions without any explicit degradation estimation.

## Architecture (design doc §4)

```
stage = [NAFBlock(s) -> PCGRMLite -> DegradationPrompt -> FiLM]
```

- `models/nafnet_blocks.py` — unmodified NAFBlock backbone primitive.
- `models/pcgrm_lite.py` — orthogonal-prototype soft cluster posterior
  (alpha), computed from the stage's global pooled feature.
- `models/degradation_prompt.py` — reparameterized degradation prompt
  (mu + sigma*eps, weighted by alpha); deterministic (eps=0) at eval.
- `models/film.py` — zero-init FiLM; starts as identity, preserves a clean
  backbone at init.
- `models/aux_order_head.py` — train-time auxiliary order classifier on the
  deepest-stage embedding. Trivially detachable; disabling it has **zero**
  effect on the restoration output path (verified by test).
- `models/cg_nafnet.py` — encoder-decoder assembly with per-stage routing.

## Layout

```
cgnafnet/
├── configs/base.yaml         # training config (single source of truth)
├── data/                     # degradation composition + datasets
├── models/                   # NAFNet + cluster routing components
├── losses/                   # charbonnier + MS-SSIM + ortho + aux losses
├── tests/                    # pytest suite (phases 1-2 checkpoints)
├── train.py                  # phase 4 training loop
├── validate_clusters.py      # phase 5 cluster-purity / t-SNE report
└── benchmark_latency.py      # phase 6 latency budget check
```

## Usage

```bash
# unit tests
python -m pytest cgnafnet/tests/

# train (full)
python -m cgnafnet.train --config cgnafnet/configs/base.yaml

# smoke run (tiny model, 2 epochs, 64px) — validates the loop end-to-end
python -m cgnafnet.train --config cgnafnet/configs/smoke.yaml

# cluster validation on a trained checkpoint
python -m cgnafnet.validate_clusters \
    --config cgnafnet/configs/base.yaml \
    --checkpoint runs/<run_id>/best.pt

# latency benchmark (default 30 ms budget)
python -m cgnafnet.benchmark_latency \
    --config cgnafnet/configs/base.yaml \
    --checkpoint runs/<run_id>/best.pt
```

## Data

Clean ground-truth images are the only real input (currently the repo's
`src/data/{train,val}/GT` grayscale npy; config accepts a list of directories
or explicit file paths so more domains can be added later). Degraded training
pairs are **synthesized on the fly** — uniformly-random order over the 6
permutations of (speckle, gaussian, downsample), continuous severities, and
downsampled results are resized back to the clean resolution (fixed-size mode,
so the model is a same-resolution restorer). The model never sees the ground
truth degradation order.

## Loss

```
L = L_recon (charbonnier) + λ_ssim·(1−MS-SSIM) + λ_ortho·‖PPᵀ−I‖_F² + λ_aux·CE(order)
```

- oracle: reconstruct clean output
- auxiliary: order-classification head (train-only, guarantees separable
  posteriors)
- orthogonality: orthogonal prototype banks per stage
- cluster entropy logged per stage: entropy stuck at max → routing isn't
  differentiating; collapse → clusters collapsing (both red flags)

## Status

- Phase 1 (data) — done, 9 unit tests
- Phase 2 (model) — done, 16 unit tests
- Phase 3 (losses) — done
- Phase 4 (training) — done, smoke run verified: loss decreases, val PSNR rises
- Phase 5 (validation) — done, t-SNE + purity report produced
- Phase 6 (latency) — done, p95 vs budget check

Definition of done per `implementation.md`: all tests green, clean forward +
backward, training smoke run converges, correct-format generated artifacts,
benchmark p95 report.