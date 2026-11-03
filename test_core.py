"""Unit tests of the torch-free modules:  pytest -q tests/test_core.py"""
import numpy as np

from intellinav3d.collision import collision_penalty
from intellinav3d.config import load_config
from intellinav3d.constraints import ConstraintChecker
from intellinav3d.encoding import augment, encode_scene, floor_mask, frame
from intellinav3d.examples import example_bedroom
from intellinav3d.geometry import obb_corners, signed_gap
from intellinav3d.graph import build_graph, compute_train_stats, undirected_edges
from intellinav3d.monff import monff_value
from intellinav3d.navigation import NavGrid, astar, dijkstra_lengths, navigation_metrics, sample_pairs
from intellinav3d.stats import holm, paired_test, summary

CFG = load_config()


def test_signed_gap():
    a = obb_corners(0, 0, 2, 2, 0)
    assert abs(signed_gap(a, obb_corners(2.5, 0, 1, 1, 0)) - 1.0) < 1e-9
    assert abs(signed_gap(a, obb_corners(1.4, 0, 1, 1, 0)) + 0.1) < 1e-9


def test_worked_example_collision_penalty():
    """Section 3.5: CP = 0.230 + 0.045 + 0.025 = 0.300 m before, 0 after correction."""
    cp, terms = collision_penalty(example_bedroom(True), CFG.collision)
    assert abs(cp - 0.300) < 1e-6 and len(terms) == 3
    assert collision_penalty(example_bedroom(False), CFG.collision)[0] == 0.0


def test_worked_example_graph():
    """Section 3.2 / Fig. 3: 8 nodes and 8 edges."""
    sc = example_bedroom(False)
    adj, rel, ef = build_graph(sc, 2.0)
    assert len(sc.objects) == 8 and len(undirected_edges(adj)) == 8


def test_monff_sample_computation():
    """Section 3.6: 0.866 before and 0.958 after refinement."""
    c1 = {"NS": 0.88, "C_s": 0.90, "A_s": 0.89, "SC": 0.94, "HC": 0.87, "CP": 0.06}
    c2 = {"NS": 0.96, "C_s": 0.97, "A_s": 0.96, "SC": 0.95, "HC": 0.95, "CP": 0.0}
    assert abs(monff_value(c1, CFG.monff) - 0.866) < 1e-9
    assert abs(monff_value(c2, CFG.monff) - 0.958) < 1e-9


def test_astar_equals_dijkstra():
    sc = example_bedroom(False)
    g = NavGrid(sc, CFG.navigation)
    for s, t in sample_pairs(sc, CFG.navigation):
        a = g.snap(g.to_cell(s), 0.5)
        b = g.snap(g.to_cell(t), 0.5)
        L1, _ = astar(g.navigable, a, b)
        L2 = dijkstra_lengths(g.navigable, a, [b])[0][0]
        assert (np.isinf(L1) and np.isinf(L2)) or abs(L1 - L2) < 1e-6


def test_navigation_metric_ranges():
    sc = example_bedroom(False)
    m = navigation_metrics(sc, CFG.navigation, sample_pairs(sc, CFG.navigation))
    for k in ("P_s", "C_s", "A_s", "ANS"):
        assert 0.0 <= m[k] <= 1.0


def test_augmentation_consistency():
    sc = example_bedroom(False)
    st = compute_train_stats([sc])
    e = encode_scene(sc, 25, st, 2.0)
    c, s = frame(sc)
    for k in range(4):
        for flip in (0, 1):
            a = augment(e, k, flip)
            fl = (np.array(sc.floor) - c) / s
            x, y = fl[:, 0].copy(), fl[:, 1].copy()
            if flip:
                x = -x
            for _ in range(k):
                x, y = -y, x
            s2 = sc.copy()
            s2.floor = (np.stack([x, y], 1) * s + c).tolist()
            assert (floor_mask(s2, 64, (c, s)) == a["floor"][0]).mean() == 1.0


def test_constraints():
    from intellinav3d.scene import Obj, Scene
    sc = Scene("t", "t", "bedroom", [[0, 0], [4, 0], [4, 4], [0, 4]],
               [Obj("table", 2.0, 0.65, 0.0, 1.0, 1.0, 0.75), Obj("dresser", 2.0, 3.75, 0.0, 1.0, 0.45, 0.8)])
    chk = ConstraintChecker(sc, CFG)
    _, ok = chk.apply(sc, 0, 0.0, -0.25, 0.0)          # would leave the room (0.15 m from the wall)
    assert not ok
    new, ok = chk.apply(sc, 1, 0.0, 0.25, 0.0)          # wall-aligned: pushed into the wall -> slides along it
    assert ok and abs(new.objects[1].y - 3.75) < 1e-9


def test_stats():
    s = summary([96.1, 96.9, 97.3, 96.2, 96.8, 96.9])
    assert s["n"] == 6 and s["ci95"] > 0
    t = paired_test([1, 2, 3, 4], [0, 1, 2, 3.5])
    assert t["mean_diff"] > 0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
