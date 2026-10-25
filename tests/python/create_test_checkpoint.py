"""Create a deterministic lightweight checkpoint for the embedded-Python C++ test."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torch

from slbjnv.model import SLbJNVModel


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: create_test_checkpoint.py OUTPUT")
    torch.manual_seed(7)
    configuration = {
        "feature_dimension": 1000,
        "embedding_size": 16,
        "hidden_size": 32,
        "key_token_features": 3,
        "lstm_layers": 1,
        "maximum_key_steps": 3,
    }
    model = SLbJNVModel(**configuration)
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": configuration}, destination)


if __name__ == "__main__":
    main()
