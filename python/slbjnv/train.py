"""Train the complete SLbJNV network on ordered Branch-and-Price processes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
from torch import nn
import yaml

from .dataset import BranchingSample, OrderedInstanceDataset, tensorize
from .loss import adaptive_score_loss, log_cosh_loss, update_beta
from .model import SLbJNVModel


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_score_loss: float
    train_sequence_loss: float
    validation_loss: float
    validation_mae: float
    validation_mse: float
    validation_sequence_loss: float
    beta: float
    learning_rate: float


def load_configuration(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    if not isinstance(configuration, dict):
        raise ValueError("configuration root must be a mapping")
    return configuration


def model_configuration(configuration: dict[str, Any]) -> dict[str, int]:
    model = configuration["model"]
    return {
        "feature_dimension": int(model["feature_dimension"]),
        "embedding_size": int(model["feature_embedding"]),
        "hidden_size": int(model["lstm_hidden"]),
        "key_token_features": int(model["key_token_features"]),
        "lstm_layers": int(model["lstm_layers"]),
        "maximum_key_steps": int(model.get("maximum_key_steps", 20)),
    }


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_batches(groups: Sequence[list[BranchingSample]], batch_size: int,
                   generator: random.Random) -> Iterable[list[list[BranchingSample]]]:
    indices = list(range(len(groups)))
    generator.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [groups[index] for index in indices[start:start + batch_size]]


def _random_samples(samples: Sequence[BranchingSample], count: int,
                    generator: random.Random) -> list[BranchingSample]:
    if not samples:
        raise ValueError("D2 random-sampling pool is empty")
    return [samples[generator.randrange(len(samples))] for _ in range(count)]


def _sequence_loss(output, batch) -> torch.Tensor:
    target = batch.key_values[:, 2:2 + output.key_sequence.predictions.shape[1]]
    return log_cosh_loss(
        output.key_sequence.predictions, target, output.key_sequence.mask
    )


def evaluate_model(model: SLbJNVModel, dataset: OrderedInstanceDataset,
                   device: torch.device) -> dict[str, float]:
    model.eval()
    absolute_sum = 0.0
    squared_sum = 0.0
    sequence_sum = 0.0
    sample_count = 0
    group_count = 0
    with torch.inference_mode():
        for group in dataset:
            batch = tensorize(group).to(device)
            output = model(batch.features, batch.key_values, batch.key_lengths, 0.0)
            errors = output.scores - batch.scores
            absolute_sum += float(errors.abs().sum())
            squared_sum += float(errors.square().sum())
            sequence_sum += float(_sequence_loss(output, batch))
            sample_count += len(group)
            group_count += 1
    if sample_count == 0:
        return {"loss": float("inf"), "mae": float("inf"),
                "mse": float("inf"), "sequence_loss": float("inf")}
    mae = absolute_sum / sample_count
    mse = squared_sum / sample_count
    sequence_loss = sequence_sum / max(group_count, 1)
    return {"loss": mse + sequence_loss, "mae": mae,
            "mse": mse, "sequence_loss": sequence_loss}


def _save_checkpoint(path: Path, model: SLbJNVModel, model_config: dict[str, int],
                     epoch: int, beta: torch.Tensor, metrics: EpochMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model_config,
        "epoch": epoch,
        "beta": float(beta),
        "metrics": asdict(metrics),
    }, path)


def _write_history(path: Path, history: Sequence[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in history)


def train(configuration: dict[str, Any], train_path: str | Path | None = None,
          validation_path: str | Path | None = None, epochs: int | None = None,
          device_name: str | None = None) -> tuple[SLbJNVModel, list[EpochMetrics]]:
    paths = configuration["paths"]
    training = configuration["training"]
    train_dataset = OrderedInstanceDataset(train_path or paths["train"])
    validation_dataset = OrderedInstanceDataset(validation_path or paths["validation"])
    if len(train_dataset) == 0:
        raise ValueError("training dataset contains no instance groups")

    seed = int(training["seed"])
    set_reproducible_seed(seed)
    generator = random.Random(seed)
    requested_device = device_name or training.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    model_config = model_configuration(configuration)
    model = SLbJNVModel(**model_config).to(device)
    optimizer_name = str(training["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training["scheduler_patience"])
    )
    beta = torch.tensor(float(training.get("initial_beta", 1.0)), device=device)
    beta_maximum = training.get("maximum_beta")
    groups = list(train_dataset)
    all_samples = train_dataset.samples
    history: list[EpochMetrics] = []
    best_validation = float("inf")
    stale_epochs = 0
    epoch_count = int(epochs if epochs is not None else training["epochs"])
    validation_for_selection = validation_dataset if len(validation_dataset) else train_dataset

    for epoch in range(1, epoch_count + 1):
        model.train()
        total_loss = total_score = total_sequence = 0.0
        batch_count = 0
        for group_batch in _group_batches(groups, int(training["batch_size"]), generator):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            batch_score = torch.zeros((), device=device)
            batch_sequence = torch.zeros((), device=device)
            random_error_means: list[torch.Tensor] = []
            for group in group_batch:
                ordered = tensorize(group).to(device)
                random_batch = tensorize(_random_samples(
                    all_samples, int(training["random_samples"]), generator
                )).to(device)
                output = model(
                    ordered.features, ordered.key_values, ordered.key_lengths,
                    float(training.get("teacher_forcing_ratio", 0.5))
                )
                random_output = model(
                    random_batch.features, random_batch.key_values,
                    random_batch.key_lengths,
                    float(training.get("teacher_forcing_ratio", 0.5))
                )
                score_loss, cumulative, random_term = adaptive_score_loss(
                    output.scores, ordered.scores, random_output.scores,
                    random_batch.scores, beta
                )
                sequence_loss = _sequence_loss(output, ordered)
                combined = score_loss + float(training.get("sequence_loss_weight", 1.0)) * sequence_loss
                batch_loss = batch_loss + combined
                batch_score = batch_score + cumulative + beta.detach() * random_term
                batch_sequence = batch_sequence + sequence_loss
                random_error_means.append(random_term.detach() / len(group))
            batch_loss = batch_loss / len(group_batch)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            beta = update_beta(
                beta, torch.stack(random_error_means).mean(),
                float(training["random_error_limit"]),
                float(training["beta_learning_rate"]),
                None if beta_maximum is None else float(beta_maximum)
            )
            total_loss += float(batch_loss.detach())
            total_score += float((batch_score / len(group_batch)).detach())
            total_sequence += float((batch_sequence / len(group_batch)).detach())
            batch_count += 1

        validation = evaluate_model(model, validation_for_selection, device)
        scheduler.step(validation["loss"])
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / batch_count,
            train_score_loss=total_score / batch_count,
            train_sequence_loss=total_sequence / batch_count,
            validation_loss=validation["loss"],
            validation_mae=validation["mae"],
            validation_mse=validation["mse"],
            validation_sequence_loss=validation["sequence_loss"],
            beta=float(beta),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        history.append(metrics)
        print(json.dumps(asdict(metrics), ensure_ascii=False))
        if metrics.validation_loss < best_validation - float(training.get("minimum_delta", 1e-6)):
            best_validation = metrics.validation_loss
            stale_epochs = 0
            _save_checkpoint(Path(paths["best_model"]), model, model_config, epoch, beta, metrics)
        else:
            stale_epochs += 1
        _save_checkpoint(Path(paths["last_model"]), model, model_config, epoch, beta, metrics)
        _write_history(Path(paths["history"]), history)
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/slbjnv.yaml"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    arguments = parser.parse_args()
    train(load_configuration(arguments.config), arguments.train, arguments.validation,
          arguments.epochs, arguments.device)


if __name__ == "__main__":
    main()
