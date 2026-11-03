"""Runs several RefineEnv instances in worker processes (numpy only); the policy stays in the main process."""
import multiprocessing as mp

import numpy as np

from ..scene import Scene
from .env import RefineEnv


def _worker(remote, cfg, stats, scene_dicts, reward_mode, constrained, seed):
    rng = np.random.default_rng(seed)
    env = RefineEnv(cfg, stats, reward_mode, constrained)
    scenes = [Scene.from_dict(d) for d in scene_dicts]

    def fresh():
        return env.reset(scenes[rng.integers(len(scenes))])

    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            remote.send(fresh())
        elif cmd == "step":
            o, r, done, info = env.step(*data)
            if done:
                info["episode_best"] = env.best
                o = fresh()
            remote.send((o, r, done, info))
        elif cmd == "close":
            remote.close()
            break


class VecEnv:
    def __init__(self, cfg, stats, scene_dicts, n_envs, reward_mode="monff", constrained=True, seed=0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        chunks = [scene_dicts[i::n_envs] or scene_dicts for i in range(n_envs)]
        for i in range(n_envs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cfg, stats, chunks[i], reward_mode, constrained, seed * 1000 + i),
                            daemon=True)
            p.start()
            b.close()
            self.remotes.append(a)
            self.procs.append(p)
        self.n = n_envs

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def step(self, ks, acts):
        for r, k, a in zip(self.remotes, ks, acts):
            r.send(("step", (int(k), np.asarray(a, float))))
        out = [r.recv() for r in self.remotes]
        obs, rews, dones, infos = zip(*out)
        return list(obs), np.array(rews, np.float32), np.array(dones, bool), list(infos)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
