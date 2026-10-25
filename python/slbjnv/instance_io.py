"""Independent reader and validator for the C++ GBD text interchange format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex


@dataclass(frozen=True)
class ItemRecord:
    identifier: int
    weight: float
    profit: float


@dataclass(frozen=True)
class MachineRecord:
    identifier: int
    capacity: float


@dataclass(frozen=True)
class GBDInstanceRecord:
    instance_id: str
    items: tuple[ItemRecord, ...]
    machines: tuple[MachineRecord, ...]
    compatibility: tuple[tuple[bool, ...], ...]

    def validate(self, require_non_singleton: bool = True) -> list[str]:
        errors: list[str] = []
        if not self.instance_id:
            errors.append("instance_id must not be empty")
        if not self.items:
            errors.append("at least one item is required")
        if not self.machines:
            errors.append("at least one machine is required")
        if [item.identifier for item in self.items] != list(range(len(self.items))):
            errors.append("item ids must be zero-based and contiguous")
        if [machine.identifier for machine in self.machines] != list(range(len(self.machines))):
            errors.append("machine ids must be zero-based and contiguous")
        if any(item.weight <= 0 or item.profit < 0 for item in self.items):
            errors.append("weights must be positive and profits nonnegative")
        if any(machine.capacity <= 0 for machine in self.machines):
            errors.append("machine capacities must be positive")
        count = len(self.items)
        if len(self.compatibility) != count or any(len(row) != count for row in self.compatibility):
            errors.append("compatibility matrix must be square")
            return errors
        if any(not self.compatibility[index][index] for index in range(count)):
            errors.append("every master must be compatible with itself")
        maximum_capacity = max((value.capacity for value in self.machines), default=0.0)
        if any(item.weight > maximum_capacity for item in self.items):
            errors.append("every item must fit on at least one machine")
        if require_non_singleton and count > 1:
            for master, master_item in enumerate(self.items):
                if not any(
                    candidate != master
                    and self.compatibility[master][candidate]
                    and master_item.weight + self.items[candidate].weight <= maximum_capacity
                    for candidate in range(count)
                ):
                    errors.append(f"master {master} has no feasible non-singleton batch")
        return errors

    def feasible_master_item_pairs(self) -> int:
        maximum_capacity = max(machine.capacity for machine in self.machines)
        return sum(
            1
            for master, master_item in enumerate(self.items)
            for candidate, candidate_item in enumerate(self.items)
            if candidate != master
            and self.compatibility[master][candidate]
            and master_item.weight + candidate_item.weight <= maximum_capacity
        )


def read_instance(path: str | Path) -> GBDInstanceRecord:
    source = Path(path)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    cursor = 0

    def take() -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError("unexpected end of file")
        value = lines[cursor]
        cursor += 1
        return value

    if take() != "SLBJNV_GBD_INSTANCE_V1":
        raise ValueError("unsupported file format")
    identity = shlex.split(take())
    if len(identity) != 2 or identity[0] != "instance_id":
        raise ValueError("invalid instance_id line")
    item_header = take().split()
    machine_header = take().split()
    if len(item_header) != 2 or item_header[0] != "num_items":
        raise ValueError("invalid num_items line")
    if len(machine_header) != 2 or machine_header[0] != "num_machines":
        raise ValueError("invalid num_machines line")
    item_count = int(item_header[1])
    machine_count = int(machine_header[1])
    if take() != "items":
        raise ValueError("missing items section")
    items = tuple(ItemRecord(int(parts[0]), float(parts[1]), float(parts[2]))
                  for parts in (take().split() for _ in range(item_count)))
    if take() != "machines":
        raise ValueError("missing machines section")
    machines = tuple(MachineRecord(int(parts[0]), float(parts[1]))
                     for parts in (take().split() for _ in range(machine_count)))
    if take() != "compatibility":
        raise ValueError("missing compatibility section")
    compatibility_rows: list[tuple[bool, ...]] = []
    for _ in range(item_count):
        row = take()
        if len(row) != item_count or any(value not in "01" for value in row):
            raise ValueError("invalid compatibility row")
        compatibility_rows.append(tuple(value == "1" for value in row))
    if take() != "end" or cursor != len(lines):
        raise ValueError("invalid end marker or trailing content")
    instance = GBDInstanceRecord(identity[1], items, machines, tuple(compatibility_rows))
    errors = instance.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return instance
