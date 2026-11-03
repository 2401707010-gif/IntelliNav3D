"""Evaluate layouts produced by another method (e.g. ATISS, DiffuScene, Nav2Scene, PhyScene re-run with
their official code on the SAME test rooms) under exactly the protocol of Section 4.1.2 and Section 4.8.

Input: a directory with one file per test room, named <scene_id>.json, either
  (a) in this repository's scene format, or
  (b) in the ATISS / DiffuScene box format:
      {"class_labels": [...category names...], "translations": [[x, y, z], ...],
       "sizes": [[sx, sy, sz] (half sizes, as in ATISS)], "angles": [[a], ...]}
      with the 3D-FRONT axis convention (y up). The floor plan and doors are taken from the test room.

    python scripts/evaluate_external.py --name ATISS --dir path/to/atiss_outputs --seed 0

Results are written like the variants of this repository (runs/outputs/external_methods/<name>/seed<k>),
so make_tables.py can add them to Table 12 (paired tests) and to Tables 3-7.
"""
import argparse
import json
import os

import numpy as np

from intellinav3d import experiments as ex
from intellinav3d.config import load_config, parse_overrides
from intellinav3d.data.front3d import map_category
from intellinav3d.data.split import load_split
from intellinav3d.geometry import wrap_angle
from intellinav3d.outputs import evaluate_output, write_json
from intellinav3d.scene import Obj, Scene, assign_supports


def convert_boxes(d, template):
    objs = []
    for lab, t, s, a in zip(d["class_labels"], d["translations"], d["sizes"], d["angles"]):
        ang = float(np.asarray(a).ravel()[0])
        objs.append(Obj(map_category(lab), float(t[0]), float(t[2]),
                        float(wrap_angle(-ang)), 2 * float(s[0]), 2 * float(s[2]), 2 * float(s[1]),
                        float(t[1] - s[1])))
    sc = Scene(template.scene_id, template.house_id, template.room_type, template.floor, objs, template.openings)
    return assign_supports(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    split = load_split(cfg.paths.split_file)
    out = os.path.join(cfg.paths.runs_dir, "outputs", "external_methods", a.name, f"seed{a.seed}")
    os.makedirs(os.path.join(out, "scenes"), exist_ok=True)
    n = 0
    for sid in split["test"]:
        p = os.path.join(a.dir, sid + ".json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        tmpl = Scene.load(os.path.join(cfg.paths.scenes_dir, sid + ".json"))
        sc = Scene.from_dict(d) if "objects" in d else convert_boxes(d, tmpl)
        sc.save(os.path.join(out, "scenes", sid + ".json"))
        n += 1
    print(f"{n} test rooms found for {a.name}")
    evaluate_output(cfg, out, a.workers, "scenes")
    ex.fid_output(cfg, out)
    write_json({"seed": a.seed, "variant": a.name}, os.path.join(out, "done.json"))
    ex.collect(cfg)


if __name__ == "__main__":
    main()
