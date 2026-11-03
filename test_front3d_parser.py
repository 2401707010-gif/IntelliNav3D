"""Parser test on a minimal house file with the 3D-FRONT JSON structure and two fake 3D-FUTURE models."""
import json
import math
import os
import tempfile

from intellinav3d.config import load_config
from intellinav3d.data.front3d import parse_house, ModelBounds


def _obj(path, sx, sy, sz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x in (-sx / 2, sx / 2):
            for y in (0, sy):
                for z in (-sz / 2, sz / 2):
                    f.write(f"v {x} {y} {z}\n")


def test_parse_house():
    d = tempfile.mkdtemp()
    fut = os.path.join(d, "future")
    _obj(os.path.join(fut, "bed01", "raw_model.obj"), 1.6, 0.5, 2.0)
    _obj(os.path.join(fut, "ns01", "raw_model.obj"), 0.45, 0.5, 0.4)
    json.dump([{"model_id": "bed01", "category": "King-size Bed"}, {"model_id": "ns01", "category": "Nightstand"}],
              open(os.path.join(fut, "model_info.json"), "w"))
    # floor 4 x 3.6 as two triangles in the x-z plane (y up)
    floor = {"uid": "f1", "type": "Floor", "xyz": [0, 0, 0, 4, 0, 0, 4, 0, 3.6, 0, 0, 3.6], "faces": [0, 1, 2, 0, 2, 3]}
    door = {"uid": "d1", "type": "Door", "xyz": [3.0, 0, 0, 3.9, 0, 0, 3.0, 2.0, 0.05, 3.9, 2.0, 0.05], "faces": []}
    q = [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)]          # 90 deg about y
    house = {"uid": "H1", "furniture": [{"uid": "u1", "jid": "bed01"}, {"uid": "u2", "jid": "ns01"}],
             "mesh": [floor, door],
             "scene": {"room": [{"type": "Bedroom", "instanceid": "Bedroom-1",
                                 "children": [{"ref": "f1"}, {"ref": "u1", "pos": [2, 0, 1.8], "rot": [0, 0, 0, 1], "scale": [1, 1, 1]},
                                              {"ref": "u2", "pos": [0.5, 0, 3.2], "rot": q, "scale": [1, 1, 1]}]}]}}
    hp = os.path.join(d, "H1.json")
    json.dump(house, open(hp, "w"))
    cfg = load_config()
    scenes = parse_house(hp, ModelBounds(fut), cfg)
    assert len(scenes) == 1
    sc = scenes[0]
    assert sc.room_type == "bedroom" and len(sc.objects) == 2
    bed = [o for o in sc.objects if o.category == "bed"][0]
    ns = [o for o in sc.objects if o.category == "nightstand"][0]
    assert abs(bed.x - 2) < 1e-6 and abs(bed.y - 1.8) < 1e-6 and abs(bed.w - 1.6) < 1e-6 and abs(bed.d - 2.0) < 1e-6
    assert abs(abs(ns.theta) - math.pi / 2) < 1e-6
    assert len(sc.floor) == 4 and abs(max(p[0] for p in sc.floor) - 4) < 1e-6
    assert len(sc.doors()) == 1 and abs(sc.doors()[0].width - 0.9) < 1e-6
    assert sc.doors()[0].normal[1] > 0.99          # pointing into the room
