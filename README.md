# IntelliNav3D: complete code

Code for *"Navigation Aware Indoor Scene Synthesis Using Graph Learning and Reinforcement-Guided Diffusion for
Autonomous Indoor Service Robots"* (Swathi & Thenmozhi). It covers every module of the paper (GAT, the
navigation-conditioned DDPM, the Adaptive Navigation Module, the Collision Map, MONFF, constrained PPO), the evaluation
protocol of Section 4.1.2 (A* + DWA trials), FID (Section 4.8), the statistics (Section 4.7), and scripts that write
**all 15 tables and all 12 figures**.

## Read this first

* **The code computes the results; it does not contain them.** Every value for IntelliNav3D in the tables is computed
  from your runs. Only the baseline rows of Tables 3–7 are fixed numbers, copied from the cited papers, as the manuscript
  itself states (`intellinav3d/published.py`). Your numbers will be close to the manuscript only if the manuscript
  numbers came from this same protocol. Report the numbers your runs produce.
* `python scripts/selftest.py` runs the entire pipeline on small procedural rooms in about 10–20 minutes. Run it
  first: if it ends with `SELFTEST PASSED`, the installation is complete.
* `MANUSCRIPT_NOTES.md` lists the places where the manuscript is ambiguous, internally inconsistent, or reports
  something the code cannot reproduce (e.g. Table 12 against published baselines, the physics-simulator row of Table 15,
  the Human Comfort definition). Please read it before submitting numbers.

## Installation

```bash
conda create -n intellinav3d python=3.10 -y && conda activate intellinav3d
pip install torch --index-url https://download.pytorch.org/whl/cu121     # match your CUDA version
pip install -r requirements.txt
pip install -e .
python -m pytest -q tests/test_core.py      # 9 unit tests (worked examples of Sections 3.2, 3.5, 3.6 included)
python scripts/selftest.py                  # end-to-end check of every stage (add --no_fid without internet)
```

## Data

Download 3D-FRONT and 3D-FUTURE-model from https://tianchi.aliyun.com/dataset/65347 and place them as

```
data/raw/3D-FRONT/*.json
data/raw/3D-FUTURE-model/model_info.json
data/raw/3D-FUTURE-model/<jid>/raw_model.obj
```

## Reproducing everything

```bash
bash run_all.sh            # or step by step:
python scripts/prepare_data.py                     # rooms -> data/scenes/*.json   (Section 3.1)
python scripts/make_split.py                       # 15,200 / 1,000 / 2,800, house level, seed 42 (Section 4.1.1)
python scripts/run_experiments.py --stage main --workers 16 --n_envs 16          # 6 seeds
python scripts/run_experiments.py --stage ablation --workers 16 --n_envs 16
python scripts/run_experiments.py --stage plausibility --workers 16 --n_envs 16
python scripts/run_experiments.py --stage heldout --workers 16 --n_envs 16
python scripts/run_experiments.py --stage sensitivity --workers 16 --n_envs 16
python scripts/make_tables.py                      # results/tables/Table1..15.csv, all_tables.md, all_tables.docx
python scripts/make_figures.py                     # results/figures/Fig1..Fig12.png (300 dpi)
```

Every trained component is cached (`runs/gat`, `runs/ddpm`, `runs/ppo`, …) under a hash of the settings that affect
it, so an interrupted command can be started again and continues where it stopped.

**Compute.** One seed of the full model on an RTX 4090 with 16 CPU cores takes roughly 3–5 hours: DDPM training
1–2 h, sampling about 20 min, PPO 0.5–1 h, refinement and evaluation 1–1.5 h. Navigation and refinement run on the
CPU, so more cores help. The complete study (6 seeds × 6 variants + 14 sensitivity settings × 6 seeds + held-out
room type) takes several days. Use `--ablation_seeds 0 1 2` or `--sensitivity_seeds 0 1 2` to reduce it.

## Where each table and figure comes from

| Output | Produced by | Source of the IntelliNav3D values |
|---|---|---|
| Table 1 | make_tables | notation (static) |
| Table 2, 10 | make_tables | `configs/default.yaml` and the split actually used |
| Table 3 | stage `main` | FID per room type, mean over seeds (`fid.json`) |
| Table 4, 5 | stage `main` | objects out of room, accessibility |
| Table 6, 7, S1 | stage `main` | navigation success / trial collision rate per room type (S1 adds kitchens) |
| Table 8 | stage `ablation` | variants `no_gat`, `no_ddpm`, `no_ppo`, `no_monff` |
| Table 9 | stage `main` | P_s, C_s, A_s, SC, HC, MONFF |
| Table 11 | stage `main` | mean, SD, t-based 95% CI over 6 seeds |
| Table 12 | `main` + `ablation` (+ `evaluate_external.py`) | paired t-tests with Holm correction |
| Table 13 | `main` + `plausibility` | before PPO / unconstrained PPO / constrained PPO |
| Table 14 | stage `sensitivity` | one factor at a time, range over settings |
| Table 15 | `main`, `heldout`, `external` | only rows that were actually run |
| Fig. 1 | make_figures | test rooms |
| Fig. 2, 8 | make_figures | architecture / workflow diagrams |
| Fig. 3, 4, 6, 7 | make_figures | worked-example bedroom of Sections 3.2–3.5 (`intellinav3d/examples.py`) |
| Fig. 5 | make_figures | trained DDPM of seed 0, reverse steps t = 1000 … 0 |
| Fig. 9, 10 | make_figures | ground truth, baselines (if evaluated) and IntelliNav3D layouts |
| Fig. 11, 12 | make_figures | published baseline values + this work |

The worked example reproduces the numbers in the text exactly: CP = 0.230 + 0.045 + 0.025 = 0.300 m before correction
and 0 after (Section 3.5), 8 nodes and 8 edges (Fig. 3), and MONFF = 0.866 → 0.958 (Section 3.6). The unit tests check
all three.

## Baselines under the same protocol (needed for a real Table 12)

Published values cannot be used in a paired test. To compare against e.g. ATISS or DiffuScene on *your* test rooms,
run their official code on the test split, export one file per room (`<scene_id>.json`, ATISS/DiffuScene box format
or this repository's format), and run

```bash
python scripts/evaluate_external.py --name ATISS --dir path/to/atiss_outputs --seed 0
```

Repeat for every seed. The method then appears in Table 12 (paired tests) and in Figs. 9–10.

## Repository layout

```
configs/default.yaml        every hyper-parameter; values not stated in the paper are marked "not in paper"
intellinav3d/
  data/front3d.py           3D-FRONT/3D-FUTURE parser (floor polygon, furniture boxes, doors/windows, supports)
  data/split.py             house-level split with near-duplicate grouping
  data/synthetic.py         procedural rooms - smoke tests only
  graph.py                  scene graph, edge features (Eq. 2), training-set distance priors
  models/gat.py             GAT with edge features (Eq. 1)
  models/ddpm.py            navigation-conditioned DDPM (Eqs. 3-8), classifier-free guidance
  navigation.py             occupancy grid, inflation, A*, P_s, C_s, A_s, ANS (Algorithm 2), HC (Eq. 16)
  dwa.py                    navigation trials: A* + Dynamic Window Approach (Section 4.1.2)
  collision.py              signed gaps (Eq. 10), CP (Eq. 11), collision map
  monff.py                  spatial consistency, MONFF (Eq. 13)
  constraints.py            Section 3.8 constraints, Table 13 plausibility statistics
  rl/env.py, rl/ppo.py      refinement environment, PPO (Eqs. 14-15), parallel environments
  pipeline.py               training / generation / refinement / selection (Algorithm 3), caching
  experiments.py            stages for Tables 3-15
  evaluate.py, fid.py, render.py, stats.py, tables.py, figures.py
scripts/                    command-line entry points (see above); render_blender.py = optional textured renderer
tests/                      unit tests and a torch-free plumbing test
```
