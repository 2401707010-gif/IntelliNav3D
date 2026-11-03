"""House-level train/validation/test split with near-duplicate control (Section 4.1.1)."""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from ..scene import Scene


def signature(sc: Scene, rnd):
    """Floor-plan geometry + object-category composition; equal signatures = near-duplicates."""
    f = sc.floor_np
    ext = np.sort(f.max(0) - f.min(0))
    dims = tuple(np.round(ext / rnd).astype(int))
    cats = tuple(sorted(Counter(o.category for o in sc.objects).items()))
    return (sc.room_type, dims, cats)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def make_split(cfg, scenes_dir=None, out=None, exclude_room_type=None):
    scenes_dir = scenes_dir or cfg.paths.scenes_dir
    out = out or cfg.paths.split_file
    sp = cfg.data.split
    files = sorted(glob.glob(os.path.join(scenes_dir, "*.json")))
    by_house = defaultdict(list)
    sig_houses = defaultdict(set)
    for p in files:
        sc = Scene.load(p)
        by_house[sc.house_id].append(sc.scene_id)
        sig_houses[signature(sc, sp.get("near_duplicate_round", 0.1))].add(sc.house_id)
    # houses that share a near-duplicate room are kept in the same partition
    uf = _UF()
    for h in by_house:
        uf.find(h)
    for hs in sig_houses.values():
        hs = sorted(hs)
        for h in hs[1:]:
            uf.union(hs[0], h)
    groups = defaultdict(list)
    for h in by_house:
        groups[uf.find(h)].append(h)
    keys = sorted(groups)
    rng = np.random.default_rng(sp.seed)
    rng.shuffle(keys)

    total = sum(len(v) for v in by_house.values())
    want = np.array([sp.n_train, sp.n_val, sp.n_test], float)
    if total < want.sum():                       # smaller dataset: keep the same proportions
        want = np.floor(want / want.sum() * total)
    n_train, n_val, n_test = want.astype(int)
    split = {"train": [], "val": [], "test": []}
    for k in keys:
        rooms = [r for h in groups[k] for r in by_house[h]]
        if len(split["test"]) < n_test:
            split["test"] += rooms
        elif len(split["val"]) < n_val:
            split["val"] += rooms
        elif len(split["train"]) < n_train:
            split["train"] += rooms
    if exclude_room_type:                       # Table 15: held-out room type
        def rtype(sid):
            return Scene.load(os.path.join(scenes_dir, sid + ".json")).room_type
        split["train"] = [s for s in split["train"] if rtype(s) != exclude_room_type]
        split["val"] = [s for s in split["val"] if rtype(s) != exclude_room_type]
        split["test"] = [s for s in split["test"] if rtype(s) == exclude_room_type]
    for k in split:
        split[k] = sorted(split[k])
    meta = {"seed": sp.seed, "n_scenes_available": total,
            "counts": {k: len(v) for k, v in split.items()}, "exclude_room_type": exclude_room_type}
    with open(out, "w") as f:
        json.dump({"meta": meta, **split}, f, indent=1)
    print("split:", meta["counts"])
    return split


def load_split(path):
    with open(path) as f:
        return json.load(f)
