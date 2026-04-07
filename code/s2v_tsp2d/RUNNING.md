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

## 7) Repro commands: baseline vs RTDL on same synthetic data

Use identical settings and data; change only use_rtdl_reward.

Baseline train:
python3 main.py -data_root ../../data/tsp2d -g_type clustered -min_n 15 -max_n 20 -use_rtdl_reward 0 -rtdl_reward_scale 1.0 -save_dir results/baseline-clustered-15-20 [other params as in run_nstep_dqn.sh]

Baseline eval:
python3 evaluate.py -data_root ../../data/tsp2d -g_type clustered -test_min_n 15 -test_max_n 20 -min_n 15 -max_n 20 -use_rtdl_reward 0 -rtdl_reward_scale 1.0 -save_dir results/baseline-clustered-15-20 [other params as in run_eval.sh]

RTDL train:
python3 main.py -data_root ../../data/tsp2d -g_type clustered -min_n 15 -max_n 20 -use_rtdl_reward 1 -rtdl_reward_scale 1.0 -save_dir results/rtdl-clustered-15-20 [same params]

RTDL eval:
python3 evaluate.py -data_root ../../data/tsp2d -g_type clustered -test_min_n 15 -test_max_n 20 -min_n 15 -max_n 20 -use_rtdl_reward 1 -rtdl_reward_scale 1.0 -save_dir results/rtdl-clustered-15-20 [same params]

Evaluate fallback: if log-*.txt is absent, evaluate.py automatically loads the latest *_iter_*.model in save_dir.
