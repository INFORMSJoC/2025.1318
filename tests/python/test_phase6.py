import copy
import json

import pytest

torch = pytest.importorskip("torch")

from slbjnv.dataset import tensorize, BranchingSample
from slbjnv.loss import adaptive_score_loss, log_cosh_loss, update_beta
from slbjnv.model import SLbJNVModel
from slbjnv.predictor import load_model, predict_scores
from slbjnv.schema import FEATURE_DIMENSION
from slbjnv.train import load_configuration, train


def sample(instance: str, index: int, values: list[float]) -> dict:
    return {
        "instance_id": instance,
        "decision_index": index,
        "node_id": index + 1,
        "parent_node_id": index,
        "branching_rule": 1,
        "branching_variable": {"item": index},
        "feature_vector": [float(index % 2)] * FEATURE_DIMENSION,
        "target_score": 1.0 + index,
        "key_node_sequence": list(range(len(values) - 2)),
        "key_value_sequence": values,
        "root_objective": 10.0,
    }


def write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_network_supports_variable_key_sequences() -> None:
    samples = [
        BranchingSample.from_mapping(sample("a", 0, [0.0, 1.0, 0.8, 1.0])),
        BranchingSample.from_mapping(sample("a", 1, [0.0, 0.9, 0.7, 0.6, 1.0])),
    ]
    batch = tensorize(samples)
    model = SLbJNVModel()
    output = model(batch.features, batch.key_values, batch.key_lengths)
    assert output.scores.shape == (2,)
    assert output.key_sequence.predictions.shape == (2, 3)
    assert output.key_sequence.mask.sum().item() == 5
    assert torch.isfinite(output.scores).all()
    predicted = model.predict(batch.features)
    assert predicted.shape == (2,)


def test_losses_and_projected_beta_update() -> None:
    predicted = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.5, 1.5])
    random_predicted = torch.tensor([0.0, 2.0], requires_grad=True)
    random_target = torch.tensor([1.0, 1.0])
    loss, cumulative, random_term = adaptive_score_loss(
        predicted, target, random_predicted, random_target, torch.tensor(1.0)
    )
    loss.backward()
    assert cumulative.item() == pytest.approx(1.0)
    assert random_term.item() > 0.0
    assert predicted.grad is not None and random_predicted.grad is not None
    assert log_cosh_loss(predicted, target).item() > 0.0
    assert update_beta(torch.tensor(1.0), torch.tensor(0.5), 0.1, 0.2).item() > 1.0
    assert update_beta(torch.tensor(0.01), torch.tensor(0.0), 1.0, 1.0).item() == 0.0


def test_one_epoch_training_checkpoint_and_prediction(tmp_path) -> None:
    train_file = tmp_path / "train.txt"
    validation_file = tmp_path / "validation.txt"
    rows = [
        sample("a", 0, [0.0, 1.0, 0.8, 1.0]),
        sample("a", 1, [0.0, 0.9, 0.7, 0.6, 1.0]),
        sample("b", 0, [0.0, 1.0, 0.75, 1.0]),
    ]
    write_jsonl(train_file, rows)
    write_jsonl(validation_file, [sample("v", 0, [0.0, 1.0, 0.9, 1.0])])
    configuration = copy.deepcopy(load_configuration("configs/slbjnv.yaml"))
    configuration["training"].update({
        "epochs": 1, "batch_size": 1, "random_samples": 2,
        "early_stopping_patience": 2, "device": "cpu",
    })
    configuration["paths"].update({
        "train": str(train_file), "validation": str(validation_file),
        "best_model": str(tmp_path / "best.pt"),
        "last_model": str(tmp_path / "last.pt"),
        "history": str(tmp_path / "history.csv"),
    })
    _, history = train(configuration)
    assert len(history) == 1
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    load_model(tmp_path / "best.pt")
    scores = predict_scores([[0.0] * FEATURE_DIMENSION, [1.0] * FEATURE_DIMENSION])
    assert len(scores) == 2
    assert all(torch.isfinite(torch.tensor(scores)))
