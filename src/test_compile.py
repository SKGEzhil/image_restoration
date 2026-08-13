"""Test whether LossCombinator works with torch.compile."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from losses import build_loss, PRESETS


def test_compile_preset(preset_name, device):
    """Try compiling the loss combinator with a given preset."""
    config = {"preset": preset_name}
    loss_fn = build_loss(config, device=device)

    try:
        compiled_loss = torch.compile(loss_fn)
    except Exception as e:
        return f"COMPILE FAILED: {type(e).__name__}: {e}"

    # Warmup + timed in one loop to catch runtime errors
    import time
    times = []
    for i in range(5):
        pred = torch.randn(2, 1, 256, 256, device=device)
        gt = torch.rand(2, 1, 256, 256, device=device)
        try:
            t0 = time.time()
            loss = compiled_loss(pred, gt)
            times.append(time.time() - t0)
        except Exception as e:
            return f"RUNTIME FAILED (iter {i}): {type(e).__name__}: {str(e)[:150]}"

    avg = sum(times) / len(times) * 1000
    return f"OK  loss={loss.item():.4f}  avg={avg:.1f}ms"


def test_eager_preset(preset_name, device):
    """Run without compile for comparison."""
    config = {"preset": preset_name}
    loss_fn = build_loss(config, device=device)

    import time
    t0 = time.time()
    for _ in range(5):
        pred = torch.randn(2, 1, 256, 256, device=device)
        gt = torch.rand(2, 1, 256, 256, device=device)
        loss = loss_fn(pred, gt)
    elapsed = time.time() - t0

    return f"eager loss={loss.item():.4f}  avg={elapsed/5*1000:.1f}ms"


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    test_presets = list(PRESETS.keys())

    print("=" * 60)
    print("EAGER (no compile)")
    print("=" * 60)
    for name in test_presets:
        result = test_eager_preset(name, device)
        print(f"  {name:30s} {result}")

    print()
    print("=" * 60)
    print("COMPILED (torch.compile)")
    print("=" * 60)
    for name in test_presets:
        result = test_compile_preset(name, device)
        print(f"  {name:30s} {result}")


if __name__ == "__main__":
    main()
