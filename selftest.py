"""End-to-end self-test (about 10-20 minutes on a GPU, longer on CPU).

Runs EVERY stage of the real pipeline - GAT, DDPM, PPO, refinement, evaluation, FID (if clean-fid can
download its Inception weights), all tables and all figures - on 160 small procedural rooms with tiny
model sizes. It checks that the code runs without errors on your machine. Its numbers mean nothing.

    python scripts/selftest.py              # with FID
    python scripts/selftest.py --no_fid     # without FID (no internet needed)
"""
import argparse
import os
import shutil
import types

WS = "selftest_workspace"

TINY = {
    "paths.scenes_dir": f"{WS}/scenes", "paths.split_file": f"{WS}/split.json", "paths.stats_file": f"{WS}/stats.json",
    "paths.runs_dir": f"{WS}/runs", "paths.results_dir": f"{WS}/results",
    "gat.hidden_dim": 32, "gat.heads": 4, "gat.epochs": 1, "gat.batch_size": 8,
    "ddpm.model_dim": 64, "ddpm.layers": 2, "ddpm.heads": 4, "ddpm.epochs": 2, "ddpm.timesteps": 50,
    "ddpm.batch_size": 8, "ddpm.candidates_per_room": 2, "ddpm.early_stop_patience": 5,
    "ppo.total_steps": 96, "ppo.rollout_steps": 48, "ppo.minibatch_size": 16, "ppo.epochs": 2, "ppo.model_dim": 32,
    "ppo.n_train_layouts": 8, "ppo.max_steps": 6, "ppo.patience": 3,
    "navigation.pairs_per_scene": 3, "fid.n_samples": 50,
    "experiments.sensitivity": {"ppo.clip": [0.1, 0.2], "graph.distance_threshold": [1.5, 2.0]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the previous workspace")
    a = ap.parse_args()
    if not a.keep:
        shutil.rmtree(WS, ignore_errors=True)

    from intellinav3d import experiments as ex
    from intellinav3d.config import load_config
    from intellinav3d.data.split import make_split
    from intellinav3d.data.synthetic import make_synthetic
    from intellinav3d.tables import build_all, write_all

    if a.no_fid:
        ex.fid_output = lambda *args, **kw: {}
    cfg = load_config(overrides=TINY)
    if not os.path.exists(cfg.paths.split_file):
        make_synthetic(cfg.paths.scenes_dir, 160, seed=0)
        make_split(cfg)
    args = types.SimpleNamespace(workers=2, n_envs=2, seeds=[0, 1], ablation_seeds=[0, 1], sensitivity_seeds=[0],
                                 external_dir=None)
    for st in ["main", "ablation", "plausibility", "sensitivity", "heldout"]:
        print(f"===== selftest stage {st} =====", flush=True)
        getattr(ex, f"stage_{st}")(cfg, args)
    import json
    ex.collect(cfg)
    res = json.load(open(os.path.join(cfg.paths.results_dir, "results.json")))
    write_all(build_all(res, cfg), os.path.join(cfg.paths.results_dir, "tables"))

    import subprocess
    import sys
    sets = [f"{k}={json.dumps(v)}" for k, v in TINY.items()]
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "make_figures.py"), "--set", *sets], check=True)
    print("\nSELFTEST PASSED - tables in", os.path.join(cfg.paths.results_dir, "tables"),
          "and figures in", os.path.join(cfg.paths.results_dir, "figures"))


if __name__ == "__main__":
    main()
