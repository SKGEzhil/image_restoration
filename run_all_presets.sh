#!/bin/bash
# Run all loss presets sequentially (Mac/Linux)
# Usage: bash run_all_presets.sh

cd "$(dirname "$0")"

PRESETS=(
  l1
  charbonnier
  l2
  ms_ssim
  log_l1
  gradient
  ffl
  tv
  l1_ssim_baseline
  char_msssim
  char_msssim_grad
  char_msssim_grad_ffl
  char_msssim_logl1
  char_msssim_logl1_ffl
  full_stack_tv
  geo_char_msssim
  geo_char_msssim_plus_ffl
  geo_char_msssim_grad
  uncert_char_msssim_grad
  uncert_full_stack
  uncert_char_msssim_logl1
)

echo "=== Running ${#PRESETS[@]} loss presets ==="
echo ""

for preset in "${PRESETS[@]}"; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="${TIMESTAMP}_${preset}"
  echo "--- [$TIMESTAMP] Starting: $preset ---"
  python src/train.py --loss-preset "$preset" --run-name "$RUN_NAME"
  echo "--- [$preset] Done ---"
  echo ""
done

echo "=== All presets complete ==="
