"""Paper-aligned sequence, cumulative, random, and adaptive loss terms."""

from __future__ import annotations

import torch
from torch import Tensor


def log_cosh_loss(predicted: Tensor, target: Tensor,
                  mask: Tensor | None = None) -> Tensor:
    difference = predicted - target
    values = difference + torch.nn.functional.softplus(-2.0 * difference) \
        - torch.log(difference.new_tensor(2.0))
    if mask is not None:
        values = values.masked_select(mask)
    if values.numel() == 0:
        return predicted.sum() * 0.0
    return values.mean()


def individual_absolute_error(predicted: Tensor, target: Tensor) -> Tensor:
    return (predicted - target).abs()


def cumulative_absolute_error(predicted: Tensor, target: Tensor) -> Tensor:
    """D1 error: preserve time order and sum across one B&B process."""
    return individual_absolute_error(predicted, target).sum()


def per_decision_random_error(predicted: Tensor, target: Tensor,
                              random_predictions: Tensor,
                              random_targets: Tensor) -> Tensor:
    """L2_t for every ordered decision using the same random D2 draw."""
    current_squared = (predicted - target).square()
    random_sum = (random_predictions - random_targets).square().sum()
    return (current_squared + random_sum) / (random_predictions.numel() + 1)


def adaptive_score_loss(sequence_predictions: Tensor, sequence_targets: Tensor,
                        random_predictions: Tensor, random_targets: Tensor,
                        beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    cumulative = cumulative_absolute_error(sequence_predictions, sequence_targets)
    random_errors = per_decision_random_error(
        sequence_predictions, sequence_targets, random_predictions, random_targets
    )
    random_term = random_errors.sum()
    return cumulative + beta.detach().clamp_min(0.0) * random_term, cumulative, random_term


@torch.no_grad()
def update_beta(beta: Tensor, mean_random_error: Tensor, random_error_limit: float,
                learning_rate: float, maximum: float | None = None) -> Tensor:
    """Projected dual ascent for Equation (8), enforcing beta >= 0."""
    updated = beta + learning_rate * (mean_random_error - random_error_limit)
    if maximum is not None:
        updated = updated.clamp_max(maximum)
    return updated.clamp_min(0.0)
