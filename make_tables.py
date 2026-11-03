"""Step 4: write Tables 1-15 (+ S1) to results/tables as CSV, Markdown and one Word file."""
import argparse
import json
import os

from intellinav3d.config import load_config, parse_overrides
from intellinav3d.tables import build_all, write_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    path = os.path.join(cfg.paths.results_dir, "results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if not res:
        print(f"warning: {path} not found or empty - tables contain only published values and '–'")
    out = os.path.join(cfg.paths.results_dir, "tables")
    write_all(build_all(res, cfg), out)
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
