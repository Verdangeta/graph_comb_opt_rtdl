#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/code/s2v_tsp2d"

# Train range (Table 2 setup)
TRAIN_MIN_N="${TRAIN_MIN_N:-50}"
TRAIN_MAX_N="${TRAIN_MAX_N:-100}"

# Test ranges from Table 2
TEST_RANGES="${TEST_RANGES:-50-100 100-200 200-300 300-400 400-500 500-600 1000-1200}"

# Shared hyperparameters (can be overridden through env vars)
NET_TYPE="${NET_TYPE:-QNet}"
DEV_ID="${DEV_ID:-0}"
N_STEP="${N_STEP:-1}"
DATA_ROOT="${DATA_ROOT:-../../data/tsp2d}"
DECAY="${DECAY:-0.1}"
RTDL_REWARD_SCALE="${RTDL_REWARD_SCALE:-1.0}"
KNN="${KNN:-10}"
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

RESULT_PREFIX="${RESULT_PREFIX:-results/table2-tsp-${G_TYPE}-train-${TRAIN_MIN_N}-${TRAIN_MAX_N}}"
BASELINE_SAVE_DIR="${BASELINE_SAVE_DIR:-${RESULT_PREFIX}/baseline}"
RTDL_SAVE_DIR="${RTDL_SAVE_DIR:-${RESULT_PREFIX}/rtdl}"

# Optional reference file for approximation ratio:
# CSV format expected:
# test_min_n,test_max_n,baseline_opt,rtdl_opt_or_shared_opt
# If shared opt is used, place same value in both opt columns.
REF_FILE="${REF_FILE:-}"

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

run_train() {
  local label="$1"
  local use_rtdl="$2"
  local save_dir="$3"
  mkdir -p "$save_dir"
  local train_log="$save_dir/log-${TRAIN_MIN_N}-${TRAIN_MAX_N}.txt"

  echo "============================================================"
  echo "[$label] TRAIN on ${TRAIN_MIN_N}-${TRAIN_MAX_N} (${G_TYPE})"
  echo "save_dir=$save_dir"
  echo "use_rtdl_reward=$use_rtdl"
  echo "============================================================"

  PYTHONUNBUFFERED=1 python3 main.py \
    -net_type "$NET_TYPE" \
    -dev_id "$DEV_ID" \
    -n_step "$N_STEP" \
    -data_root "$DATA_ROOT" \
    -decay "$DECAY" \
    -use_rtdl_reward "$use_rtdl" \
    -rtdl_reward_scale "$RTDL_REWARD_SCALE" \
    -knn "$KNN" \
    -min_n "$TRAIN_MIN_N" \
    -max_n "$TRAIN_MAX_N" \
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
    -w_scale "$W_SCALE" \
    2>&1 | tee "$train_log"

  # Persist the train-range checkpoint chosen by validation, so all test ranges
  # are evaluated on the same trained model.
  python3 - "$save_dir" "$TRAIN_MIN_N" "$TRAIN_MAX_N" <<'PY'
import glob
import os
import pathlib
import re
import sys

save_dir = pathlib.Path(sys.argv[1])
min_n = int(sys.argv[2])
max_n = int(sys.argv[3])
log_file = save_dir / f"log-{min_n}-{max_n}.txt"
best_it = -1
best_r = float("inf")
if log_file.is_file():
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "average" not in line:
            continue
        parts = line.split()
        try:
            it = int(parts[1].strip())
            r = float(parts[-1].strip())
        except Exception:
            continue
        if r < best_r:
            best_r = r
            best_it = it

model_path = None
if best_it >= 0:
    p = save_dir / f"nrange_{min_n}_{max_n}_iter_{best_it}.model"
    if p.is_file():
        model_path = p

if model_path is None:
    patt = str(save_dir / f"nrange_{min_n}_{max_n}_iter_*.model")
    cands = []
    for p in glob.glob(patt):
        m = re.search(r"_iter_(\d+)\.model$", os.path.basename(p))
        if m:
            cands.append((int(m.group(1)), p))
    if cands:
        cands.sort(key=lambda x: x[0])
        model_path = pathlib.Path(cands[-1][1])

if model_path is None:
    raise SystemExit(f"No model found in {save_dir} for train range {min_n}-{max_n}")

(save_dir / "selected_model.txt").write_text(str(model_path) + "\n", encoding="utf-8")
print(f"[train] selected model: {model_path}")
PY
}

run_eval_range() {
  local label="$1"
  local use_rtdl="$2"
  local save_dir="$3"
  local tmin="$4"
  local tmax="$5"
  mkdir -p "$save_dir/eval_ranges"
  local eval_log="$save_dir/eval_ranges/eval-${tmin}-${tmax}.log"

  echo "[$label] EVAL on ${tmin}-${tmax}"
  PYTHONUNBUFFERED=1 python3 evaluate.py \
    -net_type "$NET_TYPE" \
    -dev_id "$DEV_ID" \
    -n_step "$N_STEP" \
    -data_root "$DATA_ROOT" \
    -decay "$DECAY" \
    -use_rtdl_reward "$use_rtdl" \
    -rtdl_reward_scale "$RTDL_REWARD_SCALE" \
    -knn "$KNN" \
    -test_min_n "$tmin" \
    -test_max_n "$tmax" \
    -min_n "$TRAIN_MIN_N" \
    -max_n "$TRAIN_MAX_N" \
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

  local avg
  avg="$(extract_avg_tour_len "$eval_log")"
  if [[ -z "$avg" ]]; then
    echo "[$label] ERROR: failed to parse average tour length from $eval_log" >&2
    exit 1
  fi
  echo "$avg"
}

prepare_data_links
mkdir -p "$RESULT_PREFIX"

run_train "BASELINE" 0 "$BASELINE_SAVE_DIR"
run_train "RTDL" 1 "$RTDL_SAVE_DIR"

python3 - "$RESULT_PREFIX" "$BASELINE_SAVE_DIR" "$RTDL_SAVE_DIR" "$REF_FILE" "$TEST_RANGES" <<'PY'
import csv
import glob
import pathlib
import re
import subprocess
import sys

result_prefix = pathlib.Path(sys.argv[1])
baseline_dir = pathlib.Path(sys.argv[2])
rtdl_dir = pathlib.Path(sys.argv[3])
ref_file = sys.argv[4].strip()
test_ranges = sys.argv[5].split()

run_script = pathlib.Path("run_table2_tsp_clustered_baseline_vs_rtdl.sh")
if not run_script.exists():
    raise SystemExit("run script not found in cwd")

train_min = int(pathlib.os.environ.get("TRAIN_MIN_N", "50"))
train_max = int(pathlib.os.environ.get("TRAIN_MAX_N", "100"))
num_env = int(pathlib.os.environ.get("NUM_ENV", "1"))
mem_size = int(pathlib.os.environ.get("MEM_SIZE", "50000"))
g_type = pathlib.os.environ.get("G_TYPE", "clustered")
lr = pathlib.os.environ.get("LEARNING_RATE", "0.0001")
max_bp = int(pathlib.os.environ.get("MAX_BP_ITER", "4"))
embed = int(pathlib.os.environ.get("EMBED_DIM", "64"))
batch = int(pathlib.os.environ.get("BATCH_SIZE", "128"))
reg_hidden = int(pathlib.os.environ.get("REG_HIDDEN", "32"))
momentum = pathlib.os.environ.get("MOMENTUM", "0.9")
l2 = pathlib.os.environ.get("L2", "0.0")
w_scale = pathlib.os.environ.get("W_SCALE", "0.01")

def resolve_selected_model(save_dir: pathlib.Path) -> pathlib.Path:
    sel = save_dir / "selected_model.txt"
    if sel.is_file():
        p = pathlib.Path(sel.read_text(encoding="utf-8").strip())
        if p.is_file():
            return p

    patt = str(save_dir / f"nrange_{train_min}_{train_max}_iter_*.model")
    cands = []
    for p in glob.glob(patt):
        m = re.search(r"_iter_(\d+)\.model$", pathlib.Path(p).name)
        if m:
            cands.append((int(m.group(1)), pathlib.Path(p)))
    if not cands:
        raise SystemExit(f"No train checkpoint found in {save_dir}")
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]

baseline_model = resolve_selected_model(baseline_dir)
rtdl_model = resolve_selected_model(rtdl_dir)

def read_ref(path):
    ref = {}
    if not path:
        return ref
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[warn] REF_FILE not found: {path}; ratios will be omitted")
        return ref
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row["test_min_n"]), int(row["test_max_n"]))
            b = float(row["baseline_opt"])
            t = float(row["rtdl_opt_or_shared_opt"])
            ref[key] = (b, t)
    return ref

ref = read_ref(ref_file)

rows = []
for rr in test_ranges:
    m = re.match(r"^(\d+)-(\d+)$", rr)
    if not m:
        raise SystemExit(f"Bad range format: {rr}")
    tmin, tmax = int(m.group(1)), int(m.group(2))

    def eval_one(label, use_rtdl, save_dir, model_path):
        cmd = [
            "python3", "evaluate.py",
            "-net_type", "QNet",
            "-dev_id", "0",
            "-n_step", "1",
            "-data_root", "../../data/tsp2d",
            "-decay", "0.1",
            "-use_rtdl_reward", str(use_rtdl),
            "-rtdl_reward_scale", "1.0",
            "-knn", "10",
            "-test_min_n", str(tmin),
            "-test_max_n", str(tmax),
            "-min_n", str(train_min),
            "-max_n", str(tmax),
            "-num_env", str(num_env),
            "-max_iter", "1",
            "-mem_size", str(mem_size),
            "-g_type", g_type,
            "-learning_rate", lr,
            "-max_bp_iter", str(max_bp),
            "-save_dir", str(save_dir),
            "-embed_dim", str(embed),
            "-batch_size", str(batch),
            "-reg_hidden", str(reg_hidden),
            "-momentum", momentum,
            "-l2", l2,
            "-w_scale", w_scale,
            "-force_model", str(model_path),
        ]
        print(f"[{label}] EVAL {tmin}-{tmax}")
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        log_path = save_dir / "eval_ranges" / f"eval-{tmin}-{tmax}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        m2 = re.findall(r"average tour length:\s*([0-9eE+\-.]+)", proc.stdout + proc.stderr)
        if not m2:
            raise SystemExit(f"cannot parse average tour length for {label} {tmin}-{tmax}")
        return float(m2[-1])

    b = eval_one("BASELINE", 0, baseline_dir, baseline_model)
    r = eval_one("RTDL", 1, rtdl_dir, rtdl_model)

    ratio_b = ""
    ratio_r = ""
    if (tmin, tmax) in ref:
        rb, rr_ = ref[(tmin, tmax)]
        ratio_b = b / rb if rb else float("nan")
        ratio_r = r / rr_ if rr_ else float("nan")

    rows.append((f"{tmin}-{tmax}", b, r, ratio_b, ratio_r))

header = [
    "TestSize",
    "BaselineAvgTourLen",
    "RTDLAvgTourLen",
    "BaselineApproxRatio",
    "RTDLApproxRatio",
]
table_lines = [",".join(header)]
for size, b, r, rb, rr_ in rows:
    table_lines.append(",".join([
        size,
        f"{b:.8f}",
        f"{r:.8f}",
        (f"{rb:.6f}" if rb != "" else ""),
        (f"{rr_:.6f}" if rr_ != "" else ""),
    ]))

summary = [
    "",
    "==================== TSP(clustered) TABLE-2 STYLE ====================",
    "Training size : 50-100 (single model per method)",
    "Test sizes    : 50-100, 100-200, 200-300, 300-400, 400-500, 500-600, 1000-1200",
    "",
    "\n".join(table_lines),
    "",
]
if not ref:
    summary.append("NOTE: REF_FILE not provided -> approximation ratios are left blank.")
summary.append("======================================================================")
summary.append("")

print("\n".join(summary))

result_prefix.mkdir(parents=True, exist_ok=True)
(result_prefix / "table2_tsp_clustered_baseline_vs_rtdl.csv").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
(result_prefix / "table2_tsp_clustered_baseline_vs_rtdl.txt").write_text("\n".join(summary), encoding="utf-8")
PY
