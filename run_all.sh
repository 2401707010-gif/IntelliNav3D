#!/usr/bin/env bash
# Complete reproduction of all tables and figures (see README for time estimates).
set -euo pipefail
W=${WORKERS:-16}      # CPU processes for navigation / refinement
E=${N_ENVS:-16}       # parallel PPO environments

python scripts/prepare_data.py                       # 3D-FRONT -> data/scenes/*.json
python scripts/make_split.py                         # 15,200 / 1,000 / 2,800 house-level split, seed 42
python scripts/run_experiments.py --stage main         --workers $W --n_envs $E
python scripts/run_experiments.py --stage ablation     --workers $W --n_envs $E
python scripts/run_experiments.py --stage plausibility --workers $W --n_envs $E
python scripts/run_experiments.py --stage heldout      --workers $W --n_envs $E
python scripts/run_experiments.py --stage sensitivity  --workers $W --n_envs $E
python scripts/make_tables.py
python scripts/make_figures.py
echo "done: results/tables and results/figures"
