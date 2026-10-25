"""Evaluate a saved SLbJNV checkpoint on ordered JSON Lines samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import OrderedInstanceDataset
from .model import SLbJNVModel
from .train import evaluate_model


def load_checkpoint(path: str | Path, device: torch.device) -> SLbJNVModel:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model = SLbJNVModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    metrics = evaluate_model(
        load_checkpoint(arguments.model, device),
        OrderedInstanceDataset(arguments.data), device
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
