"""Persistent Python-side model loader and batch inference API for pybind11."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .model import SLbJNVModel
from .schema import FEATURE_DIMENSION

_model: SLbJNVModel | None = None
_device = torch.device("cpu")


def load_model(model_path: str | Path, device: str = "cpu") -> None:
    global _model, _device
    _device = torch.device(device)
    checkpoint = torch.load(Path(model_path), map_location=_device, weights_only=True)
    configuration = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model = SLbJNVModel(**configuration).to(_device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    _model = model


def predict_scores(candidate_features: Sequence[Sequence[float]],
                   key_sequence_representations: Sequence[float] | None = None) -> list[float]:
    if _model is None:
        raise RuntimeError("load_model must be called once before predict_scores")
    features = torch.as_tensor(candidate_features, dtype=torch.float32, device=_device)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"candidate_features must have shape [N, {FEATURE_DIMENSION}]")
    if key_sequence_representations is None:
        with torch.inference_mode():
            return _model.predict(features).cpu().tolist()
    else:
        representations = torch.as_tensor(key_sequence_representations, dtype=features.dtype, device=_device)
        if representations.ndim != 1 or representations.shape[0] != features.shape[0]:
            raise ValueError("key_sequence_representations must have shape [N]")
    with torch.inference_mode():
        return _model.score_from_representation(features, representations).cpu().tolist()
