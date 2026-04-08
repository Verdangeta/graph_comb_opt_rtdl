# TSP quickstart (Python 3 + optional RTDL rewards)

This project uses a C++ backend (shared library) and Python scripts as entrypoints.
For TSP, scripts are Python 3 compatible and support optional RTDL-based rewards.

## 1) Clone + submodules

```bash
git clone --recursive <repo-url>
cd graph_comb_opt
git submodule update --init --recursive
```

`third_party/RTDL_cpp` must be initialized (it is used by TSP replay code when RTDL reward is enabled).

## 2) Create conda environment (recommended)

```bash
conda env create -f ../../environment.yaml
conda activate gco-tsp
```

If you prefer manual setup, install at least:

```bash
python3 -m pip install numpy networkx tqdm
```

## 3) Build `graphnn`

```bash
cd graphnn
cp make_common.example make_common
# edit make_common for your machine (CUDA/MKL paths, USE_GPU, etc.)
make -j
```

## 4) Build TSP shared libraries

Synthetic TSP:

```bash
cd ../code/s2v_tsp2d/tsp2d_lib
cp Makefile.example Makefile
make -j
```

Real-world TSP:

```bash
cd ../../realworld_s2v_tsp2d/tsp2d_lib
cp Makefile.example Makefile
make -j
```

## 5) Train / evaluate

Synthetic:

```bash
cd ../..
cd s2v_tsp2d
./run_nstep_dqn.sh
./run_eval.sh
```

Real-world (TSPLIB):

```bash
cd ../realworld_s2v_tsp2d
./run_tsplib.sh
./run_eval.sh
```

## 6) RTDL reward switches

In TSP run scripts, the reward mode is controlled by:

- `use_rtdl_reward=0|1` (default `0`)
- `rtdl_reward_scale=<float>` (default `1.0`)

When `use_rtdl_reward=1`, replay memory rewrites per-step rewards using RTDL complexity:

`reward = -max(0, len(tour_edge) - len(matched_mst_edge)) * rtdl_reward_scale / norm`

Then standard n-step target computation is applied on top of these rewards.

<<<<<<< HEAD
## 7) One-command baseline vs RTDL comparison

Use the helper script to run two experiments (baseline and RTDL) with identical
hyperparameters, print progress, and save a final comparison summary.

```bash
cd code/s2v_tsp2d
chmod +x run_compare_baseline_rtdl.sh
./run_compare_baseline_rtdl.sh
```

By default, outputs go to:

- `results/compare-<g_type>-<min_n>-<max_n>/baseline`
- `results/compare-<g_type>-<min_n>-<max_n>/rtdl`
- summary: `results/compare-<g_type>-<min_n>-<max_n>/comparison.txt`

You can override settings via environment variables, e.g.:

```bash
MAX_ITER=1000 DEV_ID=0 MIN_N=15 MAX_N=20 ./run_compare_baseline_rtdl.sh
```

## 8) Reproduce Table-2-style TSP(clustered) comparison

This runs one training range (default `50-100`) and evaluates both baseline and
RTDL models on multiple test-size ranges (default:
`50-100,100-200,200-300,300-400,400-500,500-600,1000-1200`).

```bash
cd code/s2v_tsp2d
chmod +x run_table2_tsp_clustered_baseline_vs_rtdl.sh
./run_table2_tsp_clustered_baseline_vs_rtdl.sh
```

Main outputs:

- baseline models/results: `results/table2-tsp-clustered/train-50-100/baseline`
- rtdl models/results: `results/table2-tsp-clustered/train-50-100/rtdl`
- final report table:
  `results/table2-tsp-clustered/train-50-100/table2_tsp_clustered_compare.txt`

Optional env overrides:

```bash
TRAIN_MIN_N=50 TRAIN_MAX_N=100 MAX_ITER=200000 DEV_ID=0 \
TEST_RANGES="50-100,100-200,200-300,300-400,400-500,500-600,1000-1200" \
./run_table2_tsp_clustered_baseline_vs_rtdl.sh
```
