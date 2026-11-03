"""Step 1: convert 3D-FRONT / 3D-FUTURE into one JSON file per room (Section 3.1).

    python scripts/prepare_data.py                       # full dataset
    python scripts/prepare_data.py --synthetic 400       # procedural rooms for a smoke test only
"""
import argparse

from intellinav3d.config import load_config, parse_overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N house files")
    ap.add_argument("--synthetic", type=int, default=0, help="write N procedural rooms instead (smoke test)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.synthetic:
        from intellinav3d.data.synthetic import make_synthetic
        make_synthetic(cfg.paths.scenes_dir, a.synthetic)
        print(f"{a.synthetic} synthetic rooms written to {cfg.paths.scenes_dir} (NOT for paper results)")
    else:
        from intellinav3d.data.front3d import preprocess
        preprocess(cfg, a.limit)


if __name__ == "__main__":
    main()
