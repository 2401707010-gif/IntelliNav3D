"""Plumbing test without PyTorch: synthetic rooms -> split -> stand-in outputs (heuristic layouts, NOT the model)
-> evaluation, plausibility, collect -> all tables and figures (except Fig. 5 and FID, which need torch/clean-fid)."""
import os
import shutil
import sys

import numpy as np

from intellinav3d import experiments as ex
from intellinav3d import figures as F
from intellinav3d.config import load_config
from intellinav3d.data.split import make_split
from intellinav3d.data.synthetic import make_synthetic
from intellinav3d.graph import compute_train_stats
from intellinav3d.heuristic import heuristic_layout
from intellinav3d.outputs import evaluate_output, write_json
from intellinav3d.scene import Scene
from intellinav3d.tables import build_all, write_all


def main(ws):
    shutil.rmtree(ws, ignore_errors=True)
    over = {"paths.scenes_dir": f"{ws}/scenes", "paths.split_file": f"{ws}/split.json",
            "paths.stats_file": f"{ws}/stats.json", "paths.runs_dir": f"{ws}/runs", "paths.results_dir": f"{ws}/results"}
    cfg = load_config(overrides=over)
    make_synthetic(cfg.paths.scenes_dir, 80, seed=3)
    split = make_split(cfg)
    stats = compute_train_stats([Scene.load(f"{cfg.paths.scenes_dir}/{i}.json") for i in split["train"]])
    import json
    json.dump(stats, open(cfg.paths.stats_file, "w"))
    test = [Scene.load(f"{cfg.paths.scenes_dir}/{i}.json") for i in split["test"]]
    for variant in ["full", "no_gat", "no_ddpm", "no_ppo", "no_monff", "unconstrained"]:
        for seed in (0, 1):
            out = f"{cfg.paths.runs_dir}/outputs/main/{variant}/seed{seed}"
            rng = np.random.default_rng(seed + hash(variant) % 100)
            for d in ("scenes", "pre"):
                os.makedirs(f"{out}/{d}", exist_ok=True)
            for t in test:
                pre = heuristic_layout(t, stats, rng)
                pre.save(f"{out}/pre/{t.scene_id}.json")
                (t if variant != "no_ppo" else pre).save(f"{out}/scenes/{t.scene_id}.json")
            evaluate_output(cfg, out, 2, "scenes")
            evaluate_output(cfg, out, 2, "pre")
            ex.plausibility_output(cfg, out)
            write_json({"seed": seed}, f"{out}/done.json")
    res = json.load(open(ex.collect(cfg) and f"{cfg.paths.results_dir}/results.json"))
    tables = build_all(res, cfg)
    write_all(tables, f"{cfg.paths.results_dir}/tables")
    for name, (df, title, note) in tables.items():
        print(f"{name}: {df.shape[0]} rows")
    out = f"{cfg.paths.results_dir}/figures"
    F.fig1(cfg, test, out); F.fig2(out); F.fig3(cfg, out); F.fig4(cfg, out); F.fig6(cfg, out); F.fig7(cfg, out); F.fig8(out)
    ids = [t.scene_id for t in test][:3]
    F.fig_compare(cfg, ids, [("Ground truth", cfg.paths.scenes_dir), ("ATISS", None),
                             ("IntelliNav3D", f"{cfg.paths.runs_dir}/outputs/main/full/seed0/scenes")],
                  out, "Fig9_scene_completion.png", "test")
    F.fig11({"accessibility": 90, "objects_out_of_room": 1, "collision_rate": 3, "fid": 20}, out)
    F.fig12({"Bedroom": {"out": 1, "walk": 90, "fid": 20}}, out)
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ws_plumbing")
