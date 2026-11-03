"""OPTIONAL textured top-down renderer with Blender 4.x / Cycles (Section 4.8 mentions Blender 4.3).

The default FID path (intellinav3d/render.py) uses semantic-colour renderings and needs no Blender.
If you prefer textured renderings, render BOTH the real test rooms and the generated rooms with this script
and point the FID computation at the resulting folders.

    blender -b -P scripts/render_blender.py -- --scenes runs/outputs/main/full/seed0/scenes \
        --future data/raw/3D-FUTURE-model --out renders/gen --size 256 --samples 32

Objects without a 3D-FUTURE model (jid) are drawn as grey boxes of their bounding size.
This script runs inside Blender's Python; it was written against the Blender 4.x API.
"""
import argparse
import glob
import json
import math
import os
import sys

import bpy  # noqa: provided by Blender


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_floor(points):
    mesh = bpy.data.meshes.new("floor")
    verts = [(x, y, 0.0) for x, y in points]
    mesh.from_pydata(verts, [], [list(range(len(verts)))])
    obj = bpy.data.objects.new("floor", mesh)
    bpy.context.collection.objects.link(obj)
    mat = bpy.data.materials.new("floor_mat")
    mat.diffuse_color = (0.8, 0.8, 0.8, 1)
    obj.data.materials.append(mat)


def add_object(o, future_dir):
    path = os.path.join(future_dir, o.get("jid", ""), "raw_model.obj") if o.get("jid") else ""
    if path and os.path.exists(path):
        bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z", up_axis="Y")
        objs = list(bpy.context.selected_objects)
        bpy.ops.object.join() if len(objs) > 1 else None
        obj = bpy.context.selected_objects[0]
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.dimensions = (max(o["w"], 1e-3), max(o["d"], 1e-3), max(o["h"], 1e-3))
    obj.location = (o["x"], o["y"], o["z"] + o["h"] / 2)
    obj.rotation_euler = (0.0, 0.0, o["theta"])


def render(scene, future_dir, out_path, size, samples):
    clear()
    add_floor(scene["floor"])
    for o in scene["objects"]:
        if o["category"] in ("pendant_lamp", "ceiling_lamp"):
            continue
        add_object(o, future_dir)
    xs = [p[0] for p in scene["floor"]]
    ys = [p[1] for p in scene["floor"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ext = max(max(xs) - min(xs), max(ys) - min(ys))
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ext * 256 / (256 - 16)
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = (cx, cy, 10.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    sun = bpy.data.lights.new("sun", type="SUN")
    sun.energy = 3.0
    sun_obj = bpy.data.objects.new("sun", sun)
    sun_obj.rotation_euler = (math.radians(20), 0, 0)
    bpy.context.collection.objects.link(sun_obj)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("w")
    sc.world.color = (1, 1, 1)
    sc.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--future", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--samples", type=int, default=32)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.scenes, "*.json"))):
        with open(p) as f:
            sc = json.load(f)
        render(sc, a.future, os.path.join(a.out, sc["scene_id"] + ".png"), a.size, a.samples)


if __name__ == "__main__":
    main()
