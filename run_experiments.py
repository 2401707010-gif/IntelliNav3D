"""Step 3: train, generate, refine and evaluate (Tables 3-15).

    python scripts/run_experiments.py --stage main          # IntelliNav3D, 6 seeds  (Tables 3-7, 9, 11)
    python scripts/run_experiments.py --stage ablation      # Table 8 and Table 12
    python scripts/run_experiments.py --stage plausibility  # Table 13 (unconstrained PPO reference)
    python scripts/run_experiments.py --stage sensitivity   # Table 14
    python scripts/run_experiments.py --stage heldout       # Table 15, held-out room type
    python scripts/run_experiments.py --stage external --external_dir DIR   # Table 15, independent dataset
    python scripts/run_experiments.py --stage all

Finished components are cached, so an interrupted run can simply be started again.
"""
import argparse

from intellinav3d import experiments as ex
from intellinav3d.config import load_config, parse_overrides

STAGES = ["main", "ablation", "plausibility", "sensitivity", "heldout", "external"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default="main", choices=STAGES + ["all", "collect"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--ablation_seeds", type=int, nargs="*", default=None)
    ap.add_argument("--sensitivity_seeds", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8, help="CPU processes for navigation / refinement")
    ap.add_argument("--n_envs", type=int, default=8, help="parallel PPO environments")
    ap.add_argument("--external_dir", default=None, help="rooms of another dataset in this repo's JSON format")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    a.seeds = a.seeds if a.seeds is not None else list(cfg.experiments.seeds)
    a.ablation_seeds = a.ablation_seeds if a.ablation_seeds is not None else a.seeds
    a.sensitivity_seeds = a.sensitivity_seeds if a.sensitivity_seeds is not None else a.seeds
    stages = STAGES if a.stage == "all" else ([] if a.stage == "collect" else [a.stage])
    for st in stages:
        print(f"===== stage {st} =====", flush=True)
        getattr(ex, f"stage_{st}")(cfg, a)
    ex.collect(cfg)
    print(f"results collected in {cfg.paths.results_dir}/results.json")


if __name__ == "__main__":
    main()
