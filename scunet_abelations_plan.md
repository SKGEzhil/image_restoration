# SCUNetSR Ablation & Autoresearch Plan

## Context / Ground Truth Available Before Starting
- Baseline SCUNetSR (`dim:64`, `config:[2,2,2,2,2,2,2]`, `up_scale:2`, L1-only loss, `lr:0.0001`, `step` scheduler, `batch_size:12`, `input_resolution:128`) at 80 epochs: **SSIM ~0.762, PSNR ~26.2**, curve still rising, not flattened.
- NAFNet baseline (width32/blk8, L1+SSIM 1:0.2): SSIM ~0.755, PSNR ~25.6 at 80 epochs.
- NAFNet large (width64/blk16, L1+SSIM): SSIM 0.766 @80ep → 0.772 @200ep; PSNR plateaus ~26.1 from 80ep onward despite SSIM still climbing.
- Known behavior: SCUNet starts worse than NAFNet in early epochs but overtakes after ~30-40 epochs — convergence is slow but ceiling is higher. **Any early-stopping/short-budget decision logic must account for this — do not kill a run for being behind at low epoch count without checking its slope.**

## Global Rules for Every Experiment
1. **Fixed val cadence**: `val_every: 100` steps — keep this constant across all experiments so curves are directly comparable epoch-for-epoch / step-for-step.
2. **Log every run**: train_l1, train_ssim (if enabled), train_loss, val_l1, val_ssim, val_psnr, val_loss — at every validation point, not just final epoch.
3. **Each run must record**: final metrics, best metrics (and at which epoch/step they occurred), and the metric trend over the **last 20% of completed epochs** (slope), not just the final value.
4. **Seed fixed at 42** for all runs unless a run is specifically testing seed variance.
5. **No architecture-vs-architecture decision should be made on less than 30 epochs of data**, because of the observed late-crossover behavior. Hyperparameter-only sweeps (LR, scheduler, batch size) within the same architecture and same capacity may use short budgets (see Phase-specific epoch counts below).
6. One variable changes at a time relative to the current best-known config. Never grid-search two axes simultaneously in one phase.

---

## Phase 0 — Sanity/Calibration Run (do this first, mandatory)
**Goal:** validate that short-epoch rankings are trustworthy before relying on them for later phases.
- Run current baseline config to full 40 epochs (as set in yaml) — this becomes the reference curve for this phase's comparisons.
- Run 2 more configs (pick from Phase 1 LR candidates below) to the same 40 epochs.
- Compare metric ranking at epoch 10 vs epoch 25 vs epoch 40 for these 3 runs.
- **Decision rule:** if the ranking at epoch 10 matches the ranking at epoch 40 for at least 2 of 3 pairs, short-budget (10-15 epoch) proxies are trustworthy for Phase 1-2 (LR/scheduler) sweeps. If not, use minimum 25 epochs for those phases instead.

---

## Phase 1 — Learning Rate Sweep
**Epoch budget:** 15 epochs per run (or whatever Phase 0 validated as safe minimum).
**Configs to try:** LR = 3e-5, 1e-4 (baseline), 3e-4 — 3 runs total.
**Keep fixed:** everything else at baseline (dim:64, config:[2,2,2,2,2,2,2], step scheduler, batch_size:12).
**What to analyze after each run:**
- Val loss curve shape: does higher LR show instability/spikes (sign of too-high LR)?
- Val SSIM/PSNR at final epoch of the short run, AND the slope over the last 20% — a higher-LR run with a steeper still-rising slope may overtake a lower-LR run that plateaued early.
- Train vs val loss gap — large/growing gap signals overfitting at that LR.
**Decision rule:** pick the LR with the best combination of (a) final val_ssim, (b) positive-or-flat slope with no instability spikes in train loss. If two LRs are close, prefer the lower one (more stable, cheaper to keep tuning around later) unless the higher one shows clearly steeper improving slope.

---

## Phase 2 — Scheduler Comparison
**Epoch budget:** 15 epochs (same as Phase 1).
**Configs:** `step` (current: step_size 200000, gamma 0.5) vs `cosine`.
**Before running:** compute total training steps for the full intended epoch count (epochs × steps_per_epoch) and check whether `step_size: 200000` ever actually triggers a decay within that budget. If it doesn't, note that current "step" behavior is actually flat LR — this affects interpretation.
**Keep fixed:** best LR from Phase 1, all else at baseline.
**What to analyze:**
- Compare val_ssim/val_psnr curves — cosine typically helps more in shorter total-step budgets by decaying smoothly rather than never decaying (if step never triggers) or decaying too abruptly.
- Check val_loss stability near the end of the short run for signs either scheduler is causing oscillation.
**Decision rule:** pick scheduler with better final val_ssim/val_psnr and smoother (non-oscillating) late-run curve. This choice locks in for all subsequent phases.

---

## Phase 3 — Model Capacity (dim)
**Epoch budget:** minimum 30 epochs (per Phase 0 rule — this is an architecture-adjacent change, treat cautiously since capacity changes can shift convergence speed like the NAFNet/SCUNet crossover did).
**Configs:** `dim`: 64 (baseline) vs 96 vs 128, `config` kept at `[2,2,2,2,2,2,2]`.
**Keep fixed:** best LR + scheduler from Phase 1-2.
**What to analyze:**
- Primary: final val_ssim and val_psnr at matched epoch count.
- Slope over last 20% of epochs — larger models may still be rising at epoch 30 the way SCUNet was rising past NAFNet after epoch 30-40 in your original comparison. Do not eliminate a larger-dim run just because it's behind a smaller one at epoch 30 if its slope is clearly steeper and hasn't plateaued.
- Track wall-clock time per epoch for each dim — this feeds the inference-time/accuracy frontier decision later.
**Decision rule:** if a larger dim is still clearly rising (slope not flattening) at the epoch budget cutoff, extend that specific run to 50-60 epochs before making a final call, rather than deciding on an incomplete curve. Pick the smallest dim that gets within ~0.003 SSIM / ~0.1 PSNR of the best-performing larger dim, unless inference time budget explicitly allows the larger model.

---

## Phase 4 — Depth (config heaviness)
**Epoch budget:** minimum 30 epochs.
**Configs:** current `config:[2,2,2,2,2,2,2]` (best dim from Phase 3) vs `scunet_sr_real` heavier `[4,4,4,4,4,4,4]` (same best dim).
**Keep fixed:** best LR, scheduler, dim from Phases 1-3.
**What to analyze:** same as Phase 3 — final metrics + slope + wall-clock/epoch.
**Decision rule:** only adopt the heavier config if it shows a meaningfully better final-or-still-rising trend than the capacity gain alone justified relative to its extra epoch time cost (i.e., compare epoch-time-normalized improvement, not just raw SSIM).

---

## Phase 5 — Regularization (drop_path_rate)
**Only run this phase on whichever config wins Phase 3/4 (largest capacity config selected).**
**Epoch budget:** 30 epochs.
**Configs:** `drop_path_rate`: 0.0 (baseline) vs 0.05 vs 0.1.
**What to analyze:** primarily the train/val gap — regularization is a fix for overfitting, so if train_ssim/val_ssim gap is small at baseline, this phase is unlikely to help and can be deprioritized if time is short.
**Decision rule:** adopt only if it improves val metrics or closes a visible train/val gap without hurting val metrics.

---

## Phase 6 — Batch Size / Patch Size (lowest priority, only if time remains)
**Epoch budget:** 20 epochs.
**Configs:** batch_size 8 and 16-24 at current best LR (re-scale LR proportionally if batch size changes significantly, e.g. new_lr ≈ base_lr × (new_batch/12)). Test patch/input_resolution 160 separately, one axis at a time.
**What to analyze:** same metric set; also GPU memory headroom and time/epoch, since this phase mainly matters if Phase 3-4 models are memory-constrained.
**Decision rule:** only adopt if it clearly improves metrics at matched epoch count — this phase is exploratory, not expected to be primary lever based on current evidence.

---

## Final Step — Frontier Selection
After Phases 1-6, take every config's **(final val_ssim, final val_psnr, time/epoch, total epochs needed to plateau)** and construct a simple accuracy-vs-inference-time table. Select:
- One "best accuracy regardless of cost" config for leaderboard submission.
- One "best accuracy per unit inference time" config as the efficiency-frontier pick, in case the hackathon scores on efficiency too.

Re-run whichever final config(s) are chosen to full epoch budget (80-100) once, as a clean final training run, since all prior runs in this plan were either short-budget proxies or partial-length capacity checks.