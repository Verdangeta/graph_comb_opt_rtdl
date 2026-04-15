#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/code/s2v_tsp2d"

# Shared hyperparameters (override via env vars if needed)
NET_TYPE="${NET_TYPE:-QNet}"
DEV_ID="${DEV_ID:-0}"
N_STEP="${N_STEP:-1}"
DATA_ROOT="${DATA_ROOT:-../../data/tsp2d}"
DECAY="${DECAY:-0.1}"
RTDL_REWARD_SCALE="${RTDL_REWARD_SCALE:-1.0}"
KNN="${KNN:-10}"
MIN_N="${MIN_N:-15}"
MAX_N="${MAX_N:-20}"
TEST_MIN_N="${TEST_MIN_N:-15}"
TEST_MAX_N="${TEST_MAX_N:-20}"
NUM_ENV="${NUM_ENV:-1}"
MAX_ITER="${MAX_ITER:-200000}"
MEM_SIZE="${MEM_SIZE:-50000}"
G_TYPE="${G_TYPE:-clustered}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
MAX_BP_ITER="${MAX_BP_ITER:-4}"
EMBED_DIM="${EMBED_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
REG_HIDDEN="${REG_HIDDEN:-32}"
MOMENTUM="${MOMENTUM:-0.9}"
L2="${L2:-0.0}"
W_SCALE="${W_SCALE:-0.01}"

RESULT_PREFIX="${RESULT_PREFIX:-results/compare-${G_TYPE}-${MIN_N}-${MAX_N}}"
BASELINE_SAVE_DIR="${BASELINE_SAVE_DIR:-${RESULT_PREFIX}/baseline}"
RTDL_SAVE_DIR="${RTDL_SAVE_DIR:-${RESULT_PREFIX}/rtdl}"

TOTAL_STAGES=4
COMPLETED_STAGES=0
TOTAL_STAGE_SECONDS=0
CURRENT_STAGE_START=0

fmt_hms() {
  local sec="$1"
  local h=$((sec / 3600))
  local m=$(((sec % 3600) / 60))
  local s=$((sec % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

stage_start() {
  local title="$1"
  CURRENT_STAGE_START="$(date +%s)"
  echo
  echo ">>> [stage $((COMPLETED_STAGES + 1))/${TOTAL_STAGES}] $title"
}

stage_end() {
  local end_ts elapsed avg eta remaining
  end_ts="$(date +%s)"
  elapsed=$((end_ts - CURRENT_STAGE_START))
  COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
  TOTAL_STAGE_SECONDS=$((TOTAL_STAGE_SECONDS + elapsed))
  remaining=$((TOTAL_STAGES - COMPLETED_STAGES))
  if (( COMPLETED_STAGES > 0 )); then
    avg=$((TOTAL_STAGE_SECONDS / COMPLETED_STAGES))
  else
    avg=0
  fi
  eta=$((avg * remaining))
  echo "<<< stage done in $(fmt_hms "$elapsed"), elapsed total=$(fmt_hms "$TOTAL_STAGE_SECONDS"), est. remaining=$(fmt_hms "$eta")"
}

run_train_with_progress() {
  local max_iter="$1"
  local train_log_full="$2"
  local train_log_txt="$3"
  shift 3
  python3 - "$max_iter" "$train_log_full" "$train_log_txt" "$@" <<'PY'
import math
import re
import subprocess
import sys
import time

max_iter = int(sys.argv[1])
log_full = sys.argv[2]
log_txt = sys.argv[3]
cmd = sys.argv[4:]

iter_re = re.compile(r"\biter\s+(\d+)\b")

def fmt_hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"

def render_bar(progress: float, width: int = 28) -> str:
    p = max(0.0, min(1.0, progress))
    filled = int(round(p * width))
    return "#" * filled + "-" * (width - filled)

start = time.time()
last_iter = 0
last_line_time = start

with open(log_full, "w", encoding="utf-8") as full_f, open(log_txt, "w", encoding="utf-8") as txt_f:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        full_f.write(line)
        txt_f.write(line)
        full_f.flush()
        txt_f.flush()

        m = iter_re.search(line)
        if m:
            last_iter = int(m.group(1))
            now = time.time()
            elapsed = now - start
            progress = (last_iter / max_iter) if max_iter > 0 else 0.0
            eta = (elapsed / progress - elapsed) if progress > 0 else float("inf")
            bar = render_bar(progress)
            pct = progress * 100.0
            sys.stdout.write(
                f"[train progress] |{bar}| {pct:6.2f}% "
                f"iter={last_iter}/{max_iter} elapsed={fmt_hms(elapsed)} eta={fmt_hms(eta)}\n"
            )
            sys.stdout.flush()
            last_line_time = now

    rc = proc.wait()
    if rc != 0:
        sys.exit(rc)

total_elapsed = time.time() - start
final_progress = (last_iter / max_iter) if max_iter > 0 else 0.0
sys.stdout.write(
    f"[train progress] done: iter={last_iter}/{max_iter}, "
    f"final={final_progress * 100.0:6.2f}%, elapsed={fmt_hms(total_elapsed)}\n"
)
sys.stdout.flush()
PY
}

prepare_data_links() {
  mkdir -p "$ROOT/data/tsp2d"
  if [[ -d "$ROOT/data/train_tsp2d" ]]; then
    ln -sfn "$ROOT/data/train_tsp2d" "$ROOT/data/tsp2d/train_tsp2d"
  fi
  if [[ -d "$ROOT/data/validation_tsp2d" ]]; then
    ln -sfn "$ROOT/data/validation_tsp2d" "$ROOT/data/tsp2d/validation_tsp2d"
  fi
  if [[ -d "$ROOT/data/test_tsp2d" ]]; then
    ln -sfn "$ROOT/data/test_tsp2d" "$ROOT/data/tsp2d/test_tsp2d"
  fi
  if [[ -d "$ROOT/realworld_data/tsplib" ]]; then
    ln -sfn "$ROOT/realworld_data/tsplib" "$ROOT/data/tsplib"
  fi
}

extract_avg_tour_len() {
  local eval_log="$1"
  python3 - "$eval_log" <<'PY'
import pathlib
import re
import sys

txt = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
vals = re.findall(r"average tour length:\s*([0-9eE+\-.]+)", txt)
if not vals:
    print("")
else:
    print(vals[-1])
PY
}

run_one() {
  local label="$1"
  local use_rtdl="$2"
  local save_dir="$3"

  mkdir -p "$save_dir"
  local train_log_txt="$save_dir/log-${MIN_N}-${MAX_N}.txt"
  local train_log_full="$save_dir/train.full.log"
  local eval_log="$save_dir/eval.log"

  echo "============================================================"
  echo "[$label] training"
  echo "save_dir=$save_dir"
  echo "use_rtdl_reward=$use_rtdl"
  echo "============================================================"

  # Keep compatibility with evaluate.py (it searches log-min-max.txt)
  stage_start "${label} train"
  train_cmd=(
    python3 main.py
    -net_type "$NET_TYPE" \
    -dev_id "$DEV_ID" \
    -n_step "$N_STEP" \
    -data_root "$DATA_ROOT" \
    -decay "$DECAY" \
    -use_rtdl_reward "$use_rtdl" \
    -rtdl_reward_scale "$RTDL_REWARD_SCALE" \
    -knn "$KNN" \
    -min_n "$MIN_N" \
    -max_n "$MAX_N" \
    -num_env "$NUM_ENV" \
    -max_iter "$MAX_ITER" \
    -mem_size "$MEM_SIZE" \
    -g_type "$G_TYPE" \
    -learning_rate "$LEARNING_RATE" \
    -max_bp_iter "$MAX_BP_ITER" \
    -save_dir "$save_dir" \
    -embed_dim "$EMBED_DIM" \
    -batch_size "$BATCH_SIZE" \
    -reg_hidden "$REG_HIDDEN" \
    -momentum "$MOMENTUM" \
    -l2 "$L2" \
    -w_scale "$W_SCALE"
  )
  run_train_with_progress "$MAX_ITER" "$train_log_full" "$train_log_txt" "${train_cmd[@]}"
  stage_end

  echo "[$label] evaluating"
  stage_start "${label} eval"
  PYTHONUNBUFFERED=1 python3 evaluate.py \
    -net_type "$NET_TYPE" \
    -dev_id "$DEV_ID" \
    -n_step "$N_STEP" \
    -data_root "$DATA_ROOT" \
    -decay "$DECAY" \
    -use_rtdl_reward "$use_rtdl" \
    -rtdl_reward_scale "$RTDL_REWARD_SCALE" \
    -knn "$KNN" \
    -test_min_n "$TEST_MIN_N" \
    -test_max_n "$TEST_MAX_N" \
    -min_n "$MIN_N" \
    -max_n "$MAX_N" \
    -num_env "$NUM_ENV" \
    -max_iter 1 \
    -mem_size "$MEM_SIZE" \
    -g_type "$G_TYPE" \
    -learning_rate "$LEARNING_RATE" \
    -max_bp_iter "$MAX_BP_ITER" \
    -save_dir "$save_dir" \
    -embed_dim "$EMBED_DIM" \
    -batch_size "$BATCH_SIZE" \
    -reg_hidden "$REG_HIDDEN" \
    -momentum "$MOMENTUM" \
    -l2 "$L2" \
    -w_scale "$W_SCALE" \
    2>&1 | tee "$eval_log"
  stage_end

  local avg
  avg="$(extract_avg_tour_len "$eval_log")"
  if [[ -z "$avg" ]]; then
    echo "[$label] ERROR: failed to parse average tour length from $eval_log" >&2
    exit 1
  fi

  echo "[$label] average tour length = $avg"
  echo "$avg" > "$save_dir/avg_tour_length.txt"
}

prepare_data_links
mkdir -p "$RESULT_PREFIX"

run_one "BASELINE" 0 "$BASELINE_SAVE_DIR"
baseline_avg="$(cat "$BASELINE_SAVE_DIR/avg_tour_length.txt")"
run_one "RTDL" 1 "$RTDL_SAVE_DIR"
rtdl_avg="$(cat "$RTDL_SAVE_DIR/avg_tour_length.txt")"

python3 - "$baseline_avg" "$rtdl_avg" "$RESULT_PREFIX" "$BASELINE_SAVE_DIR" "$RTDL_SAVE_DIR" <<'PY'
import pathlib
import sys

baseline = float(sys.argv[1])
rtdl = float(sys.argv[2])
result_prefix = pathlib.Path(sys.argv[3])
baseline_dir = sys.argv[4]
rtdl_dir = sys.argv[5]

delta = rtdl - baseline
impr_pct = (baseline - rtdl) / baseline * 100.0 if baseline != 0 else float("nan")

summary = [
    "",
    "==================== FINAL COMPARISON ====================",
    f"baseline avg tour length : {baseline:.8f}",
    f"rtdl     avg tour length : {rtdl:.8f}",
    f"delta (rtdl-baseline)   : {delta:+.8f}",
    f"relative improvement     : {impr_pct:+.4f}%",
    f"baseline artifacts       : {baseline_dir}",
    f"rtdl artifacts           : {rtdl_dir}",
    "==========================================================",
    "",
]
print("\n".join(summary))

result_prefix.mkdir(parents=True, exist_ok=True)
(result_prefix / "comparison.txt").write_text("\n".join(summary), encoding="utf-8")
PY
