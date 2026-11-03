"""Pre-processing of 3D-FRONT / 3D-FUTURE into one JSON file per room (Section 3.1).

3D-FRONT house files: https://tianchi.aliyun.com/dataset/65347
Expected layout:
    <front3d_dir>/*.json                       (house files)
    <future_dir>/model_info.json
    <future_dir>/<jid>/raw_model.obj

Coordinates: 3D-FRONT is y-up. The floor plane used here is (x, z) of 3D-FRONT -> (x, y) here.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from ..geometry import points_in_polygon, polygon_area, wrap_angle
from ..scene import CEILING_CATEGORIES, Obj, Opening, Scene, assign_supports

ROOM_TYPE_MAP = {
    "bedroom": "bedroom", "masterbedroom": "bedroom", "secondbedroom": "bedroom", "kidsroom": "bedroom",
    "elderlyroom": "bedroom", "nannyroom": "bedroom",
    "livingroom": "livingroom", "livingdiningroom": "livingroom",
    "diningroom": "diningroom",
    "kitchen": "kitchen",
}

# (substring of the lower-cased 3D-FUTURE category, unified category); first match wins
CATEGORY_RULES = [
    ("kids bed", "kids_bed"), ("bunk bed", "kids_bed"), ("bed frame", "bed"), ("king-size bed", "bed"),
    ("double bed", "bed"), ("single bed", "bed"), ("couch bed", "bed"), ("nightstand", "nightstand"),
    ("wardrobe", "wardrobe"), ("drawer chest", "dresser"), ("dresser", "dresser"),
    ("dressing table", "dressing_table"), ("desk", "desk"), ("dining chair", "dining_chair"),
    ("dining table", "dining_table"), ("barstool", "stool"), ("stool", "stool"),
    ("armchair", "armchair"), ("lounge chair", "armchair"), ("hanging chair", "armchair"),
    ("chair", "chair"), ("sofa", "sofa"), ("chaise longue", "sofa"), ("loveseat", "sofa"),
    ("coffee table", "coffee_table"), ("tea table", "coffee_table"), ("corner/side table", "side_table"),
    ("round end table", "side_table"), ("side table", "side_table"), ("tv stand", "tv_stand"),
    ("bookcase", "bookshelf"), ("shelf", "shelf"), ("children cabinet", "children_cabinet"),
    ("wine cabinet", "wine_cabinet"), ("console table", "console_table"), ("sideboard", "console_table"),
    ("side cabinet", "cabinet"), ("cabinet", "cabinet"), ("pendant lamp", "pendant_lamp"),
    ("ceiling lamp", "ceiling_lamp"), ("floor lamp", "floor_lamp"), ("wall lamp", "other"),
    ("lamp", "table_lamp"), ("refrigerator", "refrigerator"), ("fridge", "refrigerator"),
    ("table", "table"),
]


def map_category(name):
    n = (name or "").lower().replace("_", " ")
    for key, cat in CATEGORY_RULES:
        if key in n:
            return cat
    return "other"


class ModelBounds:
    """Caches the local bounding box of each 3D-FUTURE model (from raw_model.obj)."""

    def __init__(self, future_dir):
        self.future_dir = future_dir
        self.cache = {}
        info_path = os.path.join(future_dir, "model_info.json")
        self.category = {}
        if os.path.exists(info_path):
            with open(info_path) as f:
                for m in json.load(f):
                    self.category[m["model_id"]] = m.get("category") or m.get("super-category") or ""

    def bounds(self, jid):
        if jid in self.cache:
            return self.cache[jid]
        path = os.path.join(self.future_dir, jid, "raw_model.obj")
        if not os.path.exists(path):
            self.cache[jid] = None
            return None
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        with open(path, errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    v = np.array(line.split()[1:4], dtype=float)
                    lo = np.minimum(lo, v)
                    hi = np.maximum(hi, v)
        self.cache[jid] = (lo, hi) if np.isfinite(lo).all() else None
        return self.cache[jid]


def quat_to_yaw(q):
    """3D-FRONT rotations are about the vertical (y) axis: q = [qx, qy, qz, qw]."""
    qx, qy, qz, qw = q
    return wrap_angle(2.0 * math.atan2(qy, qw))


def _outline_from_triangles(xyz, faces):
    """Boundary loop of a triangulated floor mesh, projected to (x, z)."""
    pts = np.asarray(xyz, float).reshape(-1, 3)[:, [0, 2]]
    key = [tuple(np.round(p, 3)) for p in pts]
    uniq = {}
    idx = np.array([uniq.setdefault(k, len(uniq)) for k in key])
    coords = np.zeros((len(uniq), 2))
    for k, i in uniq.items():
        coords[i] = k
    tris = idx[np.asarray(faces, int).reshape(-1, 3)]
    count = defaultdict(int)
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            if e[0] == e[1]:
                continue
            count[tuple(sorted(e))] += 1
    boundary = [e for e, n in count.items() if n == 1]
    if not boundary:
        return None
    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            n = nxt[0]
            if n == start:
                break
            if n in seen:
                break
            loop.append(n)
            seen.add(n)
            prev, cur = cur, n
        if len(loop) >= 3:
            loops.append(coords[loop])
    if not loops:
        return None
    poly = max(loops, key=polygon_area)
    return simplify_polygon(poly)


def simplify_polygon(poly, tol=1e-3):
    """Remove collinear vertices."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > tol:
            out.append(b)
    out = np.array(out)
    # counter-clockwise orientation
    x, y = out[:, 0], out[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:
        out = out[::-1]
    return out


def _nearest_segment(poly, p):
    best, bj, bt = np.inf, 0, 0.0
    for j in range(len(poly)):
        a, b = poly[j], poly[(j + 1) % len(poly)]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1))
        d = np.linalg.norm(p - (a + t * ab))
        if d < best:
            best, bj, bt = d, j, t
    return bj, bt, best


def _openings_for_room(poly, meshes):
    ops = []
    centroid = poly.mean(0)
    for m in meshes:
        kind = m["type"].lower()
        if "door" in kind:
            kind = "door"
        elif "window" in kind:
            kind = "window"
        else:
            continue
        pts = np.asarray(m["xyz"], float).reshape(-1, 3)[:, [0, 2]]
        if len(pts) == 0:
            continue
        c = (pts.min(0) + pts.max(0)) / 2
        j, t, dist = _nearest_segment(poly, c)
        if dist > 0.35:
            continue
        a, b = poly[j], poly[(j + 1) % len(poly)]
        u = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        proj = (pts - a) @ u
        width = float(proj.max() - proj.min())
        if width < 0.3:
            continue
        center = a + u * (proj.min() + proj.max()) / 2
        n = np.array([-u[1], u[0]])
        if np.dot(centroid - center, n) < 0:
            n = -n
        ops.append(Opening(center=center.tolist(), width=min(width, np.linalg.norm(b - a)),
                           normal=n.tolist(), kind=kind))
    return ops


def parse_house(path, models: ModelBounds, cfg):
    with open(path) as f:
        house = json.load(f)
    house_id = house.get("uid", os.path.splitext(os.path.basename(path))[0])
    furniture = {fu["uid"]: fu for fu in house.get("furniture", [])}
    meshes = {m["uid"]: m for m in house.get("mesh", [])}
    opening_meshes = [m for m in house.get("mesh", []) if any(k in m.get("type", "") for k in ("Door", "Window"))]
    scenes = []
    for room in house.get("scene", {}).get("room", []):
        rtype = ROOM_TYPE_MAP.get(room.get("type", "").lower())
        if rtype is None or rtype not in cfg.data.room_types:
            continue
        floor_xyz, floor_faces, offset = [], [], 0
        objects = []
        for ch in room.get("children", []):
            ref = ch.get("ref")
            if ref in meshes and meshes[ref].get("type") == "Floor":
                m = meshes[ref]
                floor_xyz.extend(m["xyz"])
                floor_faces.extend([f + offset for f in m["faces"]])
                offset += len(m["xyz"]) // 3
            elif ref in meshes and meshes[ref].get("type") == "Cabinet" and rtype == "kitchen":
                pts = np.asarray(meshes[ref]["xyz"], float).reshape(-1, 3)
                lo, hi = pts.min(0), pts.max(0)
                objects.append(Obj("kitchen_cabinet", float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2), 0.0,
                                   float(hi[0] - lo[0]), float(hi[2] - lo[2]), float(hi[1] - lo[1]), float(lo[1])))
            elif ref in furniture:
                fu = furniture[ref]
                jid = fu.get("jid", "")
                b = models.bounds(jid)
                if b is None:
                    continue
                lo, hi = b
                scale = np.asarray(ch.get("scale", [1, 1, 1]), float)
                pos = np.asarray(ch.get("pos", [0, 0, 0]), float)
                theta = quat_to_yaw(ch.get("rot", [0, 0, 0, 1]))
                size = (hi - lo) * scale
                center_local = (hi + lo) / 2 * scale
                c, s = math.cos(theta), math.sin(theta)
                # rotation about y (3D-FRONT): x' = c*x + s*z, z' = -s*x + c*z
                cx = pos[0] + c * center_local[0] + s * center_local[2]
                cz = pos[2] - s * center_local[0] + c * center_local[2]
                cat = map_category(models.category.get(jid) or fu.get("category") or fu.get("title", ""))
                if cat in CEILING_CATEGORIES or cat == "other":
                    continue
                # floor-plane angle: 3D-FRONT yaw about +y maps to -theta in the (x, z) plane
                objects.append(Obj(cat, float(cx), float(cz), float(wrap_angle(-theta)), float(size[0]),
                                   float(size[2]), float(size[1]), float(pos[1] + lo[1] * scale[1]), jid=jid))
        if not floor_xyz:
            continue
        poly = _outline_from_triangles(floor_xyz, floor_faces)
        if poly is None or len(poly) < 3:
            continue
        area = polygon_area(poly)
        ext = poly.max(0) - poly.min(0)
        if area < 2.0 or ext.max() > cfg.data.max_room_extent:
            continue
        inside = [o for o in objects if points_in_polygon(np.array([[o.x, o.y]]), poly)[0]]
        if not (cfg.data.min_objects <= len(inside) <= cfg.data.max_objects):
            continue
        sc = Scene(scene_id=f"{house_id}_{room.get('instanceid', len(scenes))}", house_id=house_id,
                   room_type=rtype, floor=poly.tolist(), objects=inside,
                   openings=_openings_for_room(poly, opening_meshes))
        assign_supports(sc)
        scenes.append(sc)
    return scenes


def preprocess(cfg, limit=None, workers=1):
    files = sorted(glob.glob(os.path.join(cfg.paths.front3d_dir, "*.json")))
    if limit:
        files = files[:limit]
    os.makedirs(cfg.paths.scenes_dir, exist_ok=True)
    models = ModelBounds(cfg.paths.future_dir)
    n = 0
    for i, path in enumerate(files):
        try:
            scenes = parse_house(path, models, cfg)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"skip {path}: {e}")
            continue
        for sc in scenes:
            sc.save(os.path.join(cfg.paths.scenes_dir, sc.scene_id + ".json"))
            n += 1
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] rooms written: {n}")
    print(f"done: {n} rooms")
    return n
