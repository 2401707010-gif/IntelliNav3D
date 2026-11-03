"""Step 2: house-level train / validation / test split with seed 42 (Section 4.1.1)."""
import argparse

from intellinav3d.config import load_config, parse_overrides
from intellinav3d.data.split import make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    make_split(load_config(a.config, parse_overrides(a.set)))


if __name__ == "__main__":
    main()
