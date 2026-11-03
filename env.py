"""Scene-refinement environment for PPO (Section 3.6, Section 3.8). numpy only.

State   S_t = {P_i, O_i, G_i}  (positions, orientations, graph of the current layout)
Action  A_t = (object index k, dx, dy, dtheta) with |d| <= 0.25 m, |dtheta| <= 15 deg
Reward  r_t = MONFF(S_{t+1}) - MONFF(S_t) - lambda_d * ||dp||   (invalid action: -2, scene unchanged)
Episode ends after T_max = 50 steps or when MONFF has not improved for `patience` steps.
The layout with the highest score seen in the episode is the refined output.
"""
import numpy as np

from ..constraints import ConstraintChecker
from ..encoding import encode_pose, frame, real_graph_arrays
from ..graph import node_features
from ..monff import evaluate_layout
from ..navigation import NavGrid, sample_pairs
from ..scene import ROOM2ID


class RefineEnv:
    def __init__(self, cfg, stats, reward_mode="monff", constrained=True):
        self.cfg, self.stats = cfg, stats
        self.reward_mode = reward_mode           # "monff" (proposed) or "ans" (ablation without MONFF)
        self.constrained = constrained
        self.max_n = cfg.data.max_objects
        self.thr = cfg.graph.distance_threshold

    # ------------------------------------------------------------------
    def _score(self, scene):
        return evaluate_layout(scene, self.cfg, self.stats, self.pairs, planner="dijkstra",
                               reward_mode=self.reward_mode, empty=self.empty)

    def reset(self, scene):
        self.scene = scene.copy()
        self.ref = scene.copy()
        self.pairs = sample_pairs(scene, self.cfg.navigation)
        self.empty = NavGrid(scene, self.cfg.navigation, empty=True)
        self.checker = ConstraintChecker(scene, self.cfg, constrained=self.constrained)
        self.comp = self._score(self.scene)
        self.best_scene, self.best = self.scene.copy(), self.comp["score"]
        self.best_comp = dict(self.comp)
        self.t, self.since_best = 0, 0
        return self.obs()

    def obs(self):
        sc, n, N = self.scene, len(self.scene.objects), self.max_n
        fr = frame(self.ref)
        cats, feats = node_features(sc)
        pose = encode_pose(sc, fr)
        adj, _, ef = real_graph_arrays(sc, N, self.thr)
        o = {
            "cats": np.zeros(N, np.int64), "feats": np.zeros((N, 4), np.float32), "pose": np.zeros((N, 4), np.float32),
            "mask": np.zeros(N, bool), "movable": np.zeros(N, bool), "wall": np.zeros(N, np.float32),
            "adj": adj, "ef": ef, "room": np.int64(ROOM2ID.get(sc.room_type, 0)),
            "glob": np.array([self.comp[k] for k in ("NS", "C_s", "A_s", "SC", "HC", "CP")] + [self.t / self.cfg.ppo.max_steps],
                             np.float32),
        }
        o["cats"][:n], o["feats"][:n], o["pose"][:n] = cats[:N], feats[:N], pose[:N]
        o["mask"][:n] = True
        for i in self.checker.movable:
            if i < N:
                o["movable"][i] = True
        for i in self.checker.wall:
            if i < N:
                o["wall"][i] = 1.0
        return o

    def step(self, k, a):
        """k: object index, a: array of 3 values in [-1, 1]."""
        p = self.cfg.ppo
        a = np.clip(np.asarray(a, float), -1, 1)
        dx, dy, dth = a[0] * p.max_move, a[1] * p.max_move, a[2] * np.deg2rad(p.max_rotate_deg)
        new, valid = self.checker.apply(self.scene, int(k), dx, dy, dth)
        self.t += 1
        if not valid:
            r = p.invalid_penalty
        else:
            comp = self._score(new)
            disp = float(np.hypot(new.objects[k].x - self.scene.objects[k].x, new.objects[k].y - self.scene.objects[k].y))
            lam_d = p.displacement_lambda if self.constrained else 0.0
            r = comp["score"] - self.comp["score"] - lam_d * disp
            self.scene, self.comp = new, comp
        if self.comp["score"] > self.best + 1e-6:
            self.best, self.best_scene, self.best_comp = self.comp["score"], self.scene.copy(), dict(self.comp)
            self.since_best = 0
        else:
            self.since_best += 1
        done = self.t >= p.max_steps or self.since_best >= p.patience
        return self.obs(), float(r), done, {"valid": valid}
