#!/bin/bash
# Run all loss presets in parallel with a live monitor
# Usage: bash run_all_presets.sh [MAX_PARALLEL]
#
# Environment variables:
#   MAX_PARALLEL    max concurrent workers (default: GPU count)
#   JOB_TIMEOUT     per-job timeout in seconds, 0 = no timeout (default: 0)
#   SHUTDOWN_MODE   RunPod shutdown: "stop" or "terminate" (default: stop)
#   REFRESH_SECS    monitor refresh interval (default: 3)

cd "$(dirname "$0")"

# ── Configuration ──────────────────────────────────────────────
PRESETS=(
  ms_ssim
  ffl
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

# Detect GPU count for default parallelism
if command -v nvidia-smi &>/dev/null; then
  DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
elif [[ "$(uname)" == "Darwin" ]]; then
  DETECTED_GPUS=1
else
  DETECTED_GPUS=1
fi

MAX_PARALLEL=${1:-${MAX_PARALLEL:-$DETECTED_GPUS}}
JOB_TIMEOUT=${JOB_TIMEOUT:-0}
SHUTDOWN_MODE=${SHUTDOWN_MODE:-stop}
REFRESH_SECS=${REFRESH_SECS:-3}

# RunPod — fill these in to enable auto-shutdown, leave empty to skip
RUNPOD_POD_ID=${RUNPOD_POD_ID:-""}       # e.g. "abc123-def456"
RUNPOD_API_KEY=${RUNPOD_API_KEY:-""}     # needed only for SHUTDOWN_MODE=terminate

# ── State files ────────────────────────────────────────────────
ABORTED=0

RUN_DIR=".parallel_runs_$$"
mkdir -p "$RUN_DIR"
STATUS_DIR="$RUN_DIR/status"
PID_FILE="$RUN_DIR/child_pids"
mkdir -p "$STATUS_DIR"
: > "$PID_FILE"

# ── Cleanup (EXIT handler) ─────────────────────────────────────
cleanup() {
  wait 2>/dev/null || true
  rm -rf "$RUN_DIR"
  tput cnorm 2>/dev/null || true
  stty sane 2>/dev/null || true

  if [[ "$ABORTED" -ne 1 ]]; then
    runpod_shutdown
  else
    echo ""
    echo "=== RunPod shutdown SKIPPED — pod is still running ==="
    echo "=== Re-run this script to resume or shut down manually ==="
  fi
}

# ── Signal handlers ────────────────────────────────────────────
handle_sigint() {
  ABORTED=1
  echo ""
  echo "=== Ctrl+C — killing all running jobs (pod will NOT be shut down) ==="
  # Kill all tracked child processes
  if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null || true
      # Also kill children of the child (tee, python)
      kill -- -"$pid" 2>/dev/null || true
    done < "$PID_FILE"
  fi
  # Kill any remaining background jobs
  jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
  exit 130
}

trap cleanup EXIT
trap handle_sigint INT TERM

track_pid() {
  echo "$1" >> "$PID_FILE"
}

# ── GPU query ──────────────────────────────────────────────────
query_gpu() {
  # Returns "gpu%,mem_used,mem_total" or empty string
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -1
  fi
}

# ── Elapsed time helper ────────────────────────────────────────
get_elapsed() {
  local start_time=$1
  [[ -z "$start_time" ]] && { echo "-"; return; }
  local now_s start_s diff
  now_s=$(date +%s)
  start_s=$(date -j -f "%H:%M:%S" "$start_time" +%s 2>/dev/null \
         || date -d "$start_time" +%s 2>/dev/null \
         || echo "")
  [[ -z "$start_s" ]] && { echo "-"; return; }
  diff=$((now_s - start_s))
  printf "%dh%02dm%02ds" $((diff / 3600)) $(( (diff % 3600) / 60 )) $((diff % 60))
}

# ── Worker function ────────────────────────────────────────────
run_preset() {
  local preset=$1
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_${preset}"

  echo "$run_id"          > "$STATUS_DIR/${preset}.run_id"
  echo "running"          > "$STATUS_DIR/${preset}.state"
  echo "0"                > "$STATUS_DIR/${preset}.epoch"
  echo "?"                > "$STATUS_DIR/${preset}.total_epochs"
  echo "-"                > "$STATUS_DIR/${preset}.steps_per_sec"
  echo "-"                > "$STATUS_DIR/${preset}.loss"
  echo "$(date +%H:%M:%S)" > "$STATUS_DIR/${preset}.start_time"

  local log_file="logs/log_${run_id}.log"
  mkdir -p logs

  local python_cmd="python src/train.py --loss-preset $preset --run-name $run_id"

  # Apply timeout if configured (macOS: gtimeout from brew install coreutils)
  if [[ "$JOB_TIMEOUT" -gt 0 ]]; then
    local timeout_bin="timeout"
    command -v gtimeout &>/dev/null && timeout_bin="gtimeout"
    if command -v "$timeout_bin" &>/dev/null; then
      python_cmd="$timeout_bin ${JOB_TIMEOUT} $python_cmd"
    else
      echo "  WARNING: timeout not found (install coreutils: brew install coreutils). Disabling timeout." >&2
    fi
  fi

  # Use stdbuf to force line-buffered output so tqdm \r updates become separate lines
  # macOS: gstdbuf from brew install coreutils
  local stdbuf_cmd=""
  if command -v stdbuf &>/dev/null; then
    stdbuf_cmd="stdbuf -oL"
  elif command -v gstdbuf &>/dev/null; then
    stdbuf_cmd="gstdbuf -oL"
  fi

  # Run: python | stdbuf -oL tee log | while read → parse tqdm metrics
  eval "$stdbuf_cmd $python_cmd" 2>&1 \
    | tee "$log_file" \
    | while IFS= read -r line; do
        # tqdm progress bar: Epoch X/Y: ... XX.Xstep/s ...
        if [[ "$line" =~ Epoch\ ([0-9]+)/([0-9]+) ]]; then
          echo "${BASH_REMATCH[1]}" > "$STATUS_DIR/${preset}.epoch"
          echo "${BASH_REMATCH[2]}" > "$STATUS_DIR/${preset}.total_epochs"
        fi
        # step/s from tqdm (progress bar or postfix)
        if [[ "$line" =~ ([0-9]+\.?[0-9]*)[[:space:]]*step/s ]]; then
          echo "${BASH_REMATCH[1]}" > "$STATUS_DIR/${preset}.steps_per_sec"
        fi
        # loss from tqdm postfix
        if [[ "$line" =~ loss=([0-9.]+) ]]; then
          echo "${BASH_REMATCH[1]}" > "$STATUS_DIR/${preset}.loss"
        fi
        # validation log line: [epoch X/Y step X/X global_step X] ...
        if [[ "$line" =~ \[epoch\ ([0-9]+)/([0-9]+)\ step\ ([0-9]+)/([0-9]+) ]]; then
          echo "${BASH_REMATCH[1]}" > "$STATUS_DIR/${preset}.epoch"
          echo "${BASH_REMATCH[3]}" > "$STATUS_DIR/${preset}.val_step"
        fi
      done

  local exit_code=${PIPESTATUS[0]}

  # Determine final state
  if [[ $exit_code -eq 124 && "$JOB_TIMEOUT" -gt 0 ]]; then
    echo "timeout" > "$STATUS_DIR/${preset}.state"
  elif [[ $exit_code -eq 0 ]]; then
    echo "done" > "$STATUS_DIR/${preset}.state"
  else
    echo "failed" > "$STATUS_DIR/${preset}.state"
  fi
  echo "$exit_code" > "$STATUS_DIR/${preset}.exit_code"
}

# ── Count presets in a given state ─────────────────────────────
count_state() {
  local target=$1
  local count=0
  for f in "$STATUS_DIR"/*.state; do
    [[ -f "$f" ]] || { echo 0; return; }
    [[ "$(cat "$f")" == "$target" ]] && ((count++)) || true
  done
  echo "$count"
}

# ── Monitor ────────────────────────────────────────────────────
monitor() {
  local total=${#PRESETS[@]}

  while true; do
    clear
    local running_count done_count failed_count timeout_count
    running_count=$(count_state running)
    done_count=$(count_state done)
    failed_count=$(count_state failed)
    timeout_count=$(count_state timeout)

    echo "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    printf "║  LOSS PRESET RUNNER — %-19s  Workers: %d/%d   Timeout: %s            ║\n" \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$running_count" "$MAX_PARALLEL" \
      "$( [[ $JOB_TIMEOUT -gt 0 ]] && echo "${JOB_TIMEOUT}s" || echo "none" )"
    echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    printf "║  %-22s  %-9s  %-7s  %-8s  %-10s  %-11s  %-20s ║\n" \
      "PRESET" "EPOCH" "STEP/S" "LOSS" "STATUS" "ELAPSED" "RUN_ID"
    echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"

    for preset in "${PRESETS[@]}"; do
      local state epoch total_epochs sps loss elapsed run_id
      state=$(cat "$STATUS_DIR/${preset}.state" 2>/dev/null || echo "queued")
      epoch=$(cat "$STATUS_DIR/${preset}.epoch" 2>/dev/null || echo "0")
      total_epochs=$(cat "$STATUS_DIR/${preset}.total_epochs" 2>/dev/null || echo "?")
      sps=$(cat "$STATUS_DIR/${preset}.steps_per_sec" 2>/dev/null || echo "-")
      loss=$(cat "$STATUS_DIR/${preset}.loss" 2>/dev/null || echo "-")
      run_id=$(cat "$STATUS_DIR/${preset}.run_id" 2>/dev/null || echo "-")

      local start_time
      start_time=$(cat "$STATUS_DIR/${preset}.start_time" 2>/dev/null || echo "")
      elapsed=$(get_elapsed "$start_time")

      local icon
      case "$state" in
        running)  icon="▶" ;;
        done)     icon="✔" ;;
        failed)   icon="✘" ;;
        timeout)  icon="⏱" ;;
        queued)   icon="○" ;;
        *)        icon="?" ;;
      esac

      local epoch_display
      if [[ "$state" == "running" || "$state" == "done" || "$state" == "failed" || "$state" == "timeout" ]]; then
        epoch_display="${epoch}/${total_epochs}"
      else
        epoch_display="-"
      fi

      local short_runid="${run_id:0:20}"

      printf "║ %s %-22s  %-9s  %-7s  %-8s  %-10s  %-11s  %-20s ║\n" \
        "$icon" "$preset" "$epoch_display" "$sps" "$loss" "$state" "$elapsed" "$short_runid"
    done

    # GPU row
    echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    local gpu_info
    gpu_info=$(query_gpu)
    if [[ -n "$gpu_info" ]]; then
      IFS=',' read -r gpu_util gpu_mem_used gpu_mem_total <<< "$gpu_info"
      printf "║  GPU: %s%% util | VRAM: %s/%s MB%-58s║\n" \
        "$gpu_util" "$gpu_mem_used" "$gpu_mem_total" ""
    elif [[ "$(uname)" == "Darwin" ]]; then
      printf "║  GPU: N/A (Apple Silicon / MPS)%-68s║\n" ""
    else
      printf "║  GPU: nvidia-smi not available%-70s║\n" ""
    fi

    echo "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    printf "║  ✔ %d done   ▶ %d running   ○ %d queued   ✘ %d failed   ⏱ %d timeout%-40s║\n" \
      "$done_count" "$running_count" \
      "$(($(count_state queued)))" "$failed_count" "$timeout_count" ""
    echo "║  Press Ctrl+C to abort (pod stays running)                                                       ║"
    printf "║  [refreshing every %ds — %s]%-75s║\n" \
      "$REFRESH_SECS" "$(date +%H:%M:%S)" ""
    echo "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝"

    # Check if all done
    if (( done_count + failed_count + timeout_count == total )); then
      echo ""
      echo "=== All $total presets finished: ✔ $done_count succeeded, ✘ $failed_count failed, ⏱ $timeout_count timed out ==="
      break
    fi

    sleep "$REFRESH_SECS"
  done
}

# ── RunPod shutdown ────────────────────────────────────────────
runpod_shutdown() {
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "RunPod shutdown skipped (RUNPOD_POD_ID not set)"
    return
  fi

  echo ""
  echo "=== RunPod pod shutdown ==="
  echo "  Pod ID:  $RUNPOD_POD_ID"
  echo "  Mode:    $SHUTDOWN_MODE"

  if [[ "$SHUTDOWN_MODE" == "terminate" ]]; then
    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
      echo "  ERROR: RUNPOD_API_KEY not set. Cannot terminate."
      echo "  Set RUNPOD_API_KEY or use SHUTDOWN_MODE=stop"
      return 1
    fi
    echo "  Terminating pod (stops billing)..."
    local response
    response=$(curl -s -w "\n%{http_code}" --request DELETE \
      --url "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" 2>&1)
    local http_code
    http_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | sed '$d')
    if [[ "$http_code" == "200" || "$http_code" == "204" ]]; then
      echo "  Pod terminated successfully."
    else
      echo "  ERROR: HTTP $http_code — $body"
      echo "  You may need to terminate manually from the RunPod dashboard."
    fi
  else
    echo "  Stopping pod..."
    if command -v runpodctl &>/dev/null; then
      runpodctl stop pod "$RUNPOD_POD_ID"
      echo "  Pod stopped."
    else
      echo "  ERROR: runpodctl not found. Install it or use the REST API."
      echo "  Manual stop: curl -X DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID -H 'Authorization: Bearer \$RUNPOD_API_KEY'"
    fi
  fi
}

# ── Main ───────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       LOSS PRESET PARALLEL RUNNER                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Presets:  %-44s║\n" "${#PRESETS[@]}"
printf "║  Workers:  %-44s║\n" "$MAX_PARALLEL parallel"
printf "║  GPUs:     %-44s║\n" "$DETECTED_GPUS detected"
printf "║  Timeout:  %-44s║\n" "$( [[ $JOB_TIMEOUT -gt 0 ]] && echo "${JOB_TIMEOUT}s per job" || echo "none" )"
printf "║  Shutdown: %-44s║\n" "$( [[ -n "${RUNPOD_POD_ID:-}" ]] && echo "RunPod ($SHUTDOWN_MODE)" || echo "local (no RunPod)" )"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Initialize all presets as queued
for preset in "${PRESETS[@]}"; do
  echo "queued" > "$STATUS_DIR/${preset}.state"
done

# Hide cursor during monitor
tput civis 2>/dev/null || true

# Launch monitor in background
monitor &
MONITOR_PID=$!
track_pid "$MONITOR_PID"

# ── Worker pool ────────────────────────────────────────────────
pids=()
preset_idx=0
total=${#PRESETS[@]}

run_next() {
  while (( preset_idx < total )); do
    local p="${PRESETS[$preset_idx]}"
    preset_idx=$((preset_idx + 1))
    local state
    state=$(cat "$STATUS_DIR/${p}.state" 2>/dev/null || echo "queued")
    if [[ "$state" == "queued" ]]; then
      run_preset "$p" &
      local wpid=$!
      pids+=("$wpid")
      track_pid "$wpid"
      return 0
    fi
  done
  return 1
}

# Fill initial worker slots
for (( w=0; w<MAX_PARALLEL; w++ )); do
  run_next || break
done

# Process pool: as each job finishes, start the next queued one
while (( ${#pids[@]} > 0 )); do
  new_pids=()
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      run_next || true
    else
      new_pids+=("$pid")
    fi
  done
  pids=("${new_pids[@]+"${new_pids[@]}"}")
  sleep 1
done

# Stop monitor
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

# ── Failure summary ────────────────────────────────────────────
echo ""
has_failures=false
for preset in "${PRESETS[@]}"; do
  state=$(cat "$STATUS_DIR/${preset}.state" 2>/dev/null || echo "unknown")
  if [[ "$state" == "failed" || "$state" == "timeout" ]]; then
    if ! $has_failures; then
      echo "=== Failed/timed-out presets ==="
      has_failures=true
    fi
    exit_code=$(cat "$STATUS_DIR/${preset}.exit_code" 2>/dev/null || echo "?")
    run_id=$(cat "$STATUS_DIR/${preset}.run_id" 2>/dev/null || echo "?")
    printf "  %-24s  state=%-8s  exit=%s  log=logs/log_%s.log\n" \
      "$preset" "$state" "$exit_code" "$run_id"
  fi
done
if ! $has_failures; then
  echo "=== All presets completed successfully ==="
fi
