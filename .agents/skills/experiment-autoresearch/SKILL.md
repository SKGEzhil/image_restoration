---
name: experiment-autoresearch
description: >
  Agent-driven ML experiment ablation runner. The agent reads an ablation plan
  markdown, decides what configs to try, runs training experiments by directly
  calling train.py, extracts metrics, analyzes trends, and iteratively proposes
  improved configs. Triggers on: "run ablation plan", "run ablation",
  "experiment autoresearch", "autoresearch training".
compatibility: Requires Python 3.10+, PyTorch, train.py, test.py, YAML config,
  checkpoint files (.pt), and text log files.
metadata:
  author: user
  version: "3.0"
---

# experiment-autoresearch: Agent-Driven Training Ablation

You are an autonomous ML experiment researcher. Your job is to read an ablation
plan markdown, decide what to run, execute training experiments, extract and
analyze metrics, and iteratively improve configs.

## Core Principles

1. **You are the brain.** Scripts are dumb utilities — only `extract_metrics.py`
   and `basic_stats.py` exist. YOU decide everything else.
2. **Never edit this skill or its scripts.** Do not modify `SKILL.md`,
   `extract_metrics.py`, or `basic_stats.py`. If you need custom analysis or
   scripts, create your own in the repo (e.g., `agent_analysis_script.py`).
3. **Never stream terminal output** from training. Always read log files with
   `tail` or `grep`. Terminal epoch progress consumes tokens.
3. **Poll for completion.** Check `runs/<name>/best.pt` existence + process
   status every 30-60 seconds.
4. **Detect slow runs.** Read last 5-10 log lines to compute steps/sec.
   If < 1 step/sec consistently, diagnose from context and retry with fixes.
5. **Crash recovery.** If a run fails, read the last 20 log lines to identify
   the error, apply a fix, and retry. Max 3 attempts per config.
   After 3 failures, skip and move on.
6. **Adaptive exploration.** Initial configs in the plan are seeds only.
   After each run, YOU analyze results and propose the next config.
   Up to 10-15 runs per phase.
7. **Test go/no-go.** YOU decide whether to run `test.py`.

## Workflow

When the user says "run ablation plan at <path>" or similar:

```
1. Read the ablation plan markdown file fully (use Read tool)
2. Load or create .experiment_autoresearch_state.json in repo root
3. For each phase (identified by ## Phase N headers):
   a. Identify parameter being ablated and seed configs from the plan text
   b. While runs_remaining > 0 and phase not converged:
      i.   Decide next config (from seeds, or from YOUR analysis)
      ii.  Launch training directly via python src/train.py
      iii. Poll for completion (check process + best.pt)
      iv.  Detect slow runs and crashes by reading log tail
      v.   Apply fixes and retry if needed (max 3 attempts)
      vi.  Extract metrics via extract_metrics.py
      vii. Analyze via basic_stats.py or mental math
      viii.Decide test go/no-go and run test.py directly if yes
      ix.  Update state file
   c. Pick phase winner and lock as baseline for next phase
4. After all phases: compile frontier report
```

## State File

Read and write `.experiment_autoresearch_state.json` in the repo root directly.

```json
{
  "plan_file": "scunet_abelations_plan.md",
  "current_best_config": { ... full config dict ... },
  "phases": [
    {
      "phase_id": "phase_1_lr",
      "phase_name": "Learning Rate Sweep",
      "parameter": "lr",
      "epoch_budget": 15,
      "max_runs": 12,
      "runs": [
        {
          "run_name": "scunet_p1_lr_3e-5_20250815_103022",
          "config_delta": {"lr": 0.0003},
          "status": "completed",
          "metrics": { ... from extract_metrics.py ... },
          "test_metrics": { ... from test.py, if run ... },
          "test_executed": true,
          "attempts": 1
        }
      ],
      "winner": null
    }
  ],
  "pending_phases": ["phase_1_lr", "phase_2_scheduler"],
  "completed_phases": [],
  "crashed_runs": []
}
```

## How to Read an Ablation Plan

Read the full markdown file. Identify phases from `## Phase N — <Name>` headers.

Within each phase, look for:
- **Parameter**: what is being changed (e.g. `lr`, `dim`, `scheduler.type`)
- **Epoch budget**: how many epochs per run
- **Max runs**: experiment cap for this phase
- **Seed configs**: initial values to try (from bullet lists or inline text)
- **Decision rule**: how to pick the winner
- **Analysis criteria**: what metrics to examine
- **Keep fixed**: parameters that must not change
- **Stop condition**: when to stop exploring

For nested parameters (e.g. `models.scunet_sr.dim`, `scheduler.type`),
use dot notation. You will write these as temp YAML configs.

## How to Launch Training

### Option A: Flat config changes (CLI args only)
When changing only top-level scalars like `lr`, `epochs`, `batch_size`:

```bash
cd <repo-root>
python src/train.py --run-name scunet_p1_lr_3e-5 --lr 0.0003 --epochs 15
```

### Option B: Nested config changes (temp YAML)
When changing nested params like `models.scunet_sr.dim`, `scheduler.type`:

Write a temp YAML by merging the base config with your delta:

```bash
python -c "
import yaml, copy
base = yaml.safe_load(open('src/training_config.yaml'))
delta = {'models': {'scunet_sr': {'dim': 96}}, 'scheduler': {'type': 'cosine'}}
merged = copy.deepcopy(base)
for k,v in delta.items():
    if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
        merged[k].update(v)
    else:
        merged[k] = v
yaml.dump(merged, open('src/training_config_temp_scunet_p3_dim96.yaml', 'w'))
"
python src/train.py --config src/training_config_temp_scunet_p3_dim96.yaml --run-name scunet_p3_dim96
```

The `--config` arg loads the custom YAML first, then any `--lr`, `--epochs`,
`--batch-size` CLI args override values from the YAML.

## Polling Strategy

Launch training in the background and poll:

```bash
python src/train.py --run-name <name> ... > logs/log_<name>.log 2>&1 &
TRAIN_PID=$!

while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
    # Check progress
    tail -n 5 logs/log_<name>.log
    # Check checkpoint exists
    ls runs/<name>/best.pt 2>/dev/null && echo "Checkpoint exists"
done
```

## Slow Run Detection

Every 60 seconds while polling, read last 5-10 log lines:

```bash
tail -n 10 logs/log_<name>.log
```

Find the `global_step` values and timestamps. Compute:
- `steps_per_second` = delta(global_step) / delta(time)
- If consistently < 1.0 for 3 consecutive checks → too slow

**Diagnosis:** Read last 20 lines for context.
- OOM pauses → reduce batch_size
- Model too large → reduce dim
- Compile issues → disable torch.compile (already skipped on MPS)
- Dataloader stalls → num_workers=0

**Action:** Kill the process (`kill $TRAIN_PID`), apply fix, retry.

## Crash Detection & Recovery

After the process exits, check:
1. Did `runs/<name>/best.pt` get created?
2. Read last 20 lines of `logs/log_<name>.log`

**Common errors and fixes:**

| Error in log | Fix |
|--------------|-----|
| "out of memory" / "OOM" / "CUDA" | batch_size = max(1, current // 2) |
| "NaN" / "infinity" / "inf" | lr = lr / 10.0 |
| Generic crash, no clear cause | batch_size = max(1, current // 2), num_workers=0 |
| 2nd attempt still failing | Reduce model dim by half |
| 3rd attempt | batch_size=1, num_workers=0, half dim, no compile |
| 4th attempt | Mark FAILED. Skip to next config. |

Track attempt count in the state file. Apply one fix at a time.

## Metric Extraction

After a run completes, call the dumb utility:

```bash
python .agents/skills/experiment-autoresearch/scripts/extract_metrics.py \
  --run-name <name> --repo-root . --output /tmp/<name>_metrics.json
```

Read the output JSON. Key fields you care about:
- `best_val_loss`, `best_epoch`, `best_step`
- `final_val_ssim`, `final_val_psnr`, `final_val_loss`
- `final_train_ssim`, `final_train_loss` — for train/val gap
- `val_ssim_series` — array of [step, value] for trend analysis
- `val_psnr_series`, `val_loss_series` — same format
- `train_ssim_series`, `train_loss_series`
- `num_val_points`, `num_train_points`
- `time_per_epoch_sec`
- `test_metrics` — if test was already run
- `last_log_lines` — last 10 lines for context

## Statistical Analysis

Feed a metric series to the utility:

```bash
python .agents/skills/experiment-autoresearch/scripts/basic_stats.py \
  --metrics-file /tmp/<name>_metrics.json --output /tmp/<name>_stats.json
```

Read output. Key fields:
- `slope.slope` — linear regression over last 20% of points
- `convergence.converged` — flat? (std < threshold)
- `trend` — "strongly_improving" | "improving" | "plateaued" | "declining"
- `instability.spike_count` — spikes >3x median jump

You can also do mental math: compute slope by taking last N points,
finding the linear trend. Or compare first half vs second half average.

## Test Evaluation

YOU decide whether to run test.py. Run it directly:

```bash
python src/test.py --checkpoint runs/<name>/best.pt --batch-size 8
```

Test outputs are written to:
- `runs/<name>/test_metrics.json` — aggregated metrics
- `runs/<name>/test_details.json` — per-sample metrics

**When to test:**
- Run is the phase winner so far
- Run beats baseline by > 0.005 SSIM
- Run has positive slope and train/val gap < 0.03
- Agent discretion for borderline cases

**When to skip:**
- Clearly worse than baseline
- Large train/val gap (> 0.05)
- Unstable training (many spikes)

## Phase Convergence (You Decide)

Stop a phase when any of:
1. Max runs reached (from plan)
2. Last 3 runs show no improvement in target metric
3. Parameter space exhausted (all tested values worse than best)
4. Plan explicitly says stop condition

## Adaptive Config Generation (You Decide)

Propose next config based on analysis of completed runs in the phase:

- **Bisection**: Best at A, runner-up at B → try midpoint
- **Trend extrapolation**: Values showing peak → try near-peak refinement
- **Order sweep**: If exploring unknown space, try order-of-magnitude steps
- **Rule-based**: Overfitting → reduce LR or add regularization.
  Flat metrics → increase LR or capacity. Still rising → extend epochs.

## References

- `references/METRICS_GUIDE.md` — detailed metric interpretation

## Gotchas

- `train.py` supports `--config <path>` for custom YAML configs.
- CLI args (`--lr`, `--epochs`, `--batch-size`) override YAML values.
- Wandb is optional. Set `WANDB_DISABLED=true` to avoid hangs.
- `runs/<name>/best.pt` is the primary metric source. Logs are secondary.
- Never read full log files into context. Use `tail -n N` or `grep` only.
- `src/test.py --checkpoint runs/<name>/best.pt` evaluates on test set.
