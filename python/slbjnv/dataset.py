"""JSONL dataset loading with instance-level grouping and stable decision order."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor

from .schema import FEATURE_DIMENSION


@dataclass(frozen=True)
class BranchingSample:
    instance_id: str
    decision_index: int
    node_id: int
    parent_node_id: int
    branching_rule: int
    branching_variable: dict
    feature_vector: tuple[float, ...]
    target_score: float
    key_node_sequence: tuple[int, ...]
    key_value_sequence: tuple[float, ...]
    root_objective: float

    @classmethod
    def from_mapping(cls, raw: dict) -> "BranchingSample":
        features = tuple(float(value) for value in raw["feature_vector"])
        if len(features) != FEATURE_DIMENSION:
            raise ValueError(f"feature_vector must have {FEATURE_DIMENSION} values")
        key_nodes = tuple(int(value) for value in raw["key_node_sequence"])
        key_values = tuple(float(value) for value in raw["key_value_sequence"])
        if len(key_values) != len(key_nodes) + 2:
            raise ValueError("key_value_sequence must contain start/end tokens around all key nodes")
        return cls(
            instance_id=str(raw["instance_id"]),
            decision_index=int(raw["decision_index"]),
            node_id=int(raw["node_id"]),
            parent_node_id=int(raw["parent_node_id"]),
            branching_rule=int(raw["branching_rule"]),
            branching_variable=dict(raw["branching_variable"]),
            feature_vector=features,
            target_score=float(raw["target_score"]),
            key_node_sequence=key_nodes,
            key_value_sequence=key_values,
            root_objective=float(raw["root_objective"]),
        )


def read_jsonl(path: str | Path) -> list[BranchingSample]:
    records: list[BranchingSample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(BranchingSample.from_mapping(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid sample at {path}:{line_number}: {exc}") from exc
    return records


def group_ordered(samples: Iterable[BranchingSample]) -> list[list[BranchingSample]]:
    grouped: dict[str, list[BranchingSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.instance_id, []).append(sample)
    result: list[list[BranchingSample]] = []
    for instance_id in sorted(grouped):
        sequence = sorted(grouped[instance_id], key=lambda value: value.decision_index)
        indices = [value.decision_index for value in sequence]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate decision_index in instance {instance_id}")
        result.append(sequence)
    return result


class OrderedInstanceDataset(Sequence[list[BranchingSample]]):
    """D1-style groups; group order may shuffle, within-group order may not."""

    def __init__(self, path: str | Path):
        self._groups = group_ordered(read_jsonl(path))

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> list[BranchingSample]:
        return self._groups[index]

    def __iter__(self) -> Iterator[list[BranchingSample]]:
        return iter(self._groups)

    @property
    def samples(self) -> list[BranchingSample]:
        return [sample for group in self._groups for sample in group]


@dataclass
class TensorBatch:
    features: Tensor
    scores: Tensor
    key_values: Tensor
    key_lengths: Tensor

    def to(self, device: torch.device | str) -> "TensorBatch":
        return TensorBatch(
            self.features.to(device), self.scores.to(device),
            self.key_values.to(device), self.key_lengths.to(device)
        )


def tensorize(samples: Sequence[BranchingSample]) -> TensorBatch:
    if not samples:
        raise ValueError("cannot tensorize an empty sample sequence")
    lengths = torch.tensor([len(sample.key_value_sequence) for sample in samples], dtype=torch.long)
    maximum = int(lengths.max().item())
    key_values = torch.zeros((len(samples), maximum), dtype=torch.float32)
    for row, sample in enumerate(samples):
        values = torch.tensor(sample.key_value_sequence, dtype=torch.float32)
        key_values[row, :values.numel()] = values
    return TensorBatch(
        torch.tensor([sample.feature_vector for sample in samples], dtype=torch.float32),
        torch.tensor([sample.target_score for sample in samples], dtype=torch.float32),
        key_values,
        lengths,
    )
