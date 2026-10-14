"""Validate one or more generated GBD TXT files independently of the C++ reader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from slbjnv.instance_io import read_instance


def collect_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir():
            paths.extend(sorted(value.glob("*.txt")))
        else:
            paths.append(value)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    paths = collect_paths(arguments.paths)
    if not paths:
        parser.error("no instance files found")
    failures = 0
    config = None
    if arguments.config:
        config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    for path in paths:
        try:
            instance = read_instance(path)
            if config:
                weight_min, weight_max = config["weight_range"]
                profit_min, profit_max = config["profit_range"]
                capacity_min, capacity_max = config["capacity_range"]
                if any(not weight_min <= item.weight <= weight_max for item in instance.items):
                    raise ValueError("item weight is outside configured range")
                if any(not profit_min <= item.profit <= profit_max for item in instance.items):
                    raise ValueError("item profit is outside configured range")
                if any(not capacity_min <= machine.capacity <= capacity_max for machine in instance.machines):
                    raise ValueError("machine capacity is outside configured range")
            print(f"OK {path}: machines={len(instance.machines)} items={len(instance.items)} "
                  f"feasible_pairs={instance.feasible_master_item_pairs()}")
        except Exception as error:
            failures += 1
            print(f"INVALID {path}: {error}", file=sys.stderr)
    print(f"validated={len(paths)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
