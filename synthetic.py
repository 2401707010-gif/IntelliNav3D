"""Procedural rooms in the 3D-FRONT style.

FOR SMOKE TESTS ONLY. These rooms let the whole pipeline run end-to-end in minutes without the
3D-FRONT download. They are not used for any result reported in the paper.
"""
import math
import os

import numpy as np

from ..scene import Obj, Opening, Scene, assign_supports


def _rect(W, H):
    return [[0, 0], [W, 0], [W, H], [0, H]]


def _lshape(W, H, cw, ch):
    return [[0, 0], [W, 0], [W, H - ch], [W - cw, H - ch], [W - cw, H], [0, H]]


def _against(rng, W, H, wall, w, d, along=None):
    """Centre and angle for an object of size (w, d) with its back against a wall of a W x H room."""
    gap = rng.uniform(0.0, 0.03)
    if wall == 0:   # bottom wall (y = 0), facing +y
        x = along if along is not None else rng.uniform(w / 2 + 0.1, W - w / 2 - 0.1)
        return x, d / 2 + gap, 0.0
    if wall == 1:   # right wall, facing -x
        y = along if along is not None else rng.uniform(w / 2 + 0.1, H - w / 2 - 0.1)
        return W - d / 2 - gap, y, math.pi / 2
    if wall == 2:   # top wall, facing -y
        x = along if along is not None else rng.uniform(w / 2 + 0.1, W - w / 2 - 0.1)
        return x, H - d / 2 - gap, math.pi
    y = along if along is not None else rng.uniform(w / 2 + 0.1, H - w / 2 - 0.1)
    return d / 2 + gap, y, -math.pi / 2


def _add(objs, cat, x, y, th, w, d, h, z=0.0):
    objs.append(Obj(cat, float(x), float(y), float(th), float(w), float(d), float(h), float(z)))


def _door(rng, W, H):
    wall = int(rng.integers(0, 4))
    width = 0.9
    if wall == 0:
        return Opening([float(rng.uniform(0.6, W - 0.6)), 0.0], width, [0.0, 1.0], "door")
    if wall == 1:
        return Opening([W, float(rng.uniform(0.6, H - 0.6))], width, [-1.0, 0.0], "door")
    if wall == 2:
        return Opening([float(rng.uniform(0.6, W - 0.6)), H], width, [0.0, -1.0], "door")
    return Opening([0.0, float(rng.uniform(0.6, H - 0.6))], width, [1.0, 0.0], "door")


def bedroom(rng, sid):
    W, H = rng.uniform(3.4, 4.6), rng.uniform(3.2, 4.2)
    objs = []
    bw, bd = rng.uniform(1.5, 1.9), rng.uniform(2.0, 2.2)
    bx = rng.uniform(1.4, W - 1.4)
    x, y, th = _against(rng, W, H, 2, bw, bd, along=bx)
    _add(objs, "bed", x, y, th, bw, bd, 0.5)
    for side in (-1, 1):
        nx = bx + side * (bw / 2 + 0.25)
        _add(objs, "nightstand", nx, H - 0.22, math.pi, 0.45, 0.4, 0.5)
    _add(objs, "table_lamp", bx + bw / 2 + 0.25, H - 0.22, 0.0, 0.25, 0.25, 0.4, 0.5)
    x, y, th = _against(rng, W, H, 1, 1.2, 0.6)
    _add(objs, "wardrobe", x, y, th, 1.2, 0.6, 2.1)
    x, y, th = _against(rng, W, H, 3, 1.1, 0.55)
    _add(objs, "desk", x, y, th, 1.1, 0.55, 0.75)
    _add(objs, "chair", x + 0.55, y, th, 0.45, 0.45, 0.9)
    if rng.random() < 0.6:
        x, y, th = _against(rng, W, H, 0, 1.0, 0.45)
        _add(objs, "dresser", x, y, th, 1.0, 0.45, 0.8)
    return Scene(sid, sid.split("_")[0], "bedroom", _rect(W, H), objs, [_door(rng, W, H)])


def livingroom(rng, sid):
    W, H = rng.uniform(4.5, 6.5), rng.uniform(3.8, 5.5)
    floor = _rect(W, H) if rng.random() < 0.7 else _lshape(W, H, 1.2, 1.0)
    objs = []
    sw = rng.uniform(2.0, 2.6)
    x, y, th = _against(rng, W, H, 0, sw, 0.9, along=W / 2)
    _add(objs, "sofa", x, y, th, sw, 0.9, 0.85)
    _add(objs, "coffee_table", W / 2, 0.9 + 0.55, 0.0, 1.1, 0.6, 0.45)
    x, y, th = _against(rng, W, H, 2, 1.8, 0.45, along=W / 2)
    _add(objs, "tv_stand", x, min(y, H - 1.25), th, 1.8, 0.45, 0.5)
    _add(objs, "armchair", W / 2 + sw / 2 + 0.6, 1.4, math.pi / 2, 0.8, 0.8, 0.85)
    _add(objs, "side_table", W / 2 - sw / 2 - 0.35, 0.3, 0.0, 0.45, 0.45, 0.55)
    if rng.random() < 0.7:
        x, y, th = _against(rng, W, H, 3, 0.9, 0.35)
        _add(objs, "bookshelf", x, y, th, 0.9, 0.35, 1.8)
    _add(objs, "floor_lamp", 0.35, H - 1.4, 0.0, 0.35, 0.35, 1.6)
    return Scene(sid, sid.split("_")[0], "livingroom", floor, objs, [_door(rng, W, H)])


def diningroom(rng, sid):
    W, H = rng.uniform(3.4, 4.8), rng.uniform(3.2, 4.4)
    objs = []
    tw, td = rng.uniform(1.4, 1.8), rng.uniform(0.8, 0.95)
    cx, cy = W / 2 + rng.uniform(-0.2, 0.2), H / 2 + rng.uniform(-0.2, 0.2)
    _add(objs, "dining_table", cx, cy, 0.0, tw, td, 0.75)
    for k in (-1, 1):
        for s in (-1, 1):
            _add(objs, "dining_chair", cx + k * tw / 4, cy + s * (td / 2 + 0.3), 0.0 if s < 0 else math.pi, 0.45, 0.45, 0.9)
    x, y, th = _against(rng, W, H, 1, 1.4, 0.45)
    _add(objs, "console_table", x, y, th, 1.4, 0.45, 0.85)
    if rng.random() < 0.5:
        x, y, th = _against(rng, W, H, 3, 0.8, 0.4)
        _add(objs, "wine_cabinet", x, y, th, 0.8, 0.4, 1.8)
    return Scene(sid, sid.split("_")[0], "diningroom", _rect(W, H), objs, [_door(rng, W, H)])


def kitchen(rng, sid):
    W, H = rng.uniform(2.8, 4.0), rng.uniform(2.6, 3.6)
    objs = []
    x, y, th = _against(rng, W, H, 2, W - 0.8, 0.6, along=(W - 0.8) / 2 + 0.05)
    _add(objs, "kitchen_cabinet", x, y, th, W - 0.8, 0.6, 0.9)
    x, y, th = _against(rng, W, H, 3, 1.4, 0.6, along=1.0)
    _add(objs, "kitchen_cabinet", x, y, th, 1.4, 0.6, 0.9)
    x, y, th = _against(rng, W, H, 1, 0.7, 0.7, along=H - 0.5)
    _add(objs, "refrigerator", x, y, th, 0.7, 0.7, 1.8)
    if rng.random() < 0.5:
        _add(objs, "table", W / 2 + 0.3, H / 2 - 0.5, 0.0, 0.9, 0.7, 0.75)
    return Scene(sid, sid.split("_")[0], "kitchen", _rect(W, H), objs, [_door(rng, W, H)])


GENERATORS = {"bedroom": bedroom, "livingroom": livingroom, "diningroom": diningroom, "kitchen": kitchen}


def jitter(rng, scene, pos=0.08, ang=0.08):
    lo, hi = scene.bounds()
    for o in scene.objects:
        if o.parent >= 0:
            continue
        o.x += rng.normal(0, pos)
        o.y += rng.normal(0, pos)
        o.theta += rng.normal(0, ang)
        c = o.corners()                      # push back inside the room bounding box
        o.x += max(0.0, lo[0] - c[:, 0].min()) - max(0.0, c[:, 0].max() - hi[0])
        o.y += max(0.0, lo[1] - c[:, 1].min()) - max(0.0, c[:, 1].max() - hi[1])
    for o in scene.objects:
        if o.parent >= 0:
            p = scene.objects[o.parent]
            o.x, o.y = p.x, p.y
    return scene


def make_synthetic(out_dir, n_rooms=200, seed=0, rooms_per_house=2):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    types = list(GENERATORS)
    for i in range(n_rooms):
        house = f"syn{i // rooms_per_house:05d}"
        rt = types[i % len(types)]
        sid = f"{house}_{rt}{i}"
        sc = GENERATORS[rt](rng, sid)
        assign_supports(sc)
        jitter(rng, sc)
        sc.save(os.path.join(out_dir, sid + ".json"))
    return n_rooms
