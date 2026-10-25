"""Combined embedding, key-sequence, and score networks for SLbJNV."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .key_sequence_network import KeySequenceNetwork, KeySequenceOutput
from .schema import FEATURE_DIMENSION, FEATURE_SLICES
from .score_network import ScoreNetwork


@dataclass
class ModelOutput:
    scores: Tensor
    key_sequence: KeySequenceOutput


class SLbJNVModel(nn.Module):
    def __init__(self, feature_dimension: int = FEATURE_DIMENSION, embedding_size: int = 16,
                 hidden_size: int = 32, key_token_features: int = 3,
                 lstm_layers: int = 1, maximum_key_steps: int = 20):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.feature_embedding = nn.Linear(feature_dimension, embedding_size)
        self.key_sequence = KeySequenceNetwork(
            key_token_features, embedding_size, hidden_size, lstm_layers, maximum_key_steps
        )
        self.score = ScoreNetwork(embedding_size + 1, hidden_size)

    def score_from_representation(self, features: Tensor,
                                  key_sequence_representation: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dimension:
            raise ValueError(f"features must have shape [N, {self.feature_dimension}]")
        embedded = self.feature_embedding(features)
        return self.score(torch.cat((embedded, key_sequence_representation.unsqueeze(-1)), dim=-1))

    def forward(self, features: Tensor, key_values: Tensor, lengths: Tensor,
                teacher_forcing_ratio: float = 1.0) -> ModelOutput:
        key_output = self.key_sequence(key_values, lengths, teacher_forcing_ratio)
        return ModelOutput(
            self.score_from_representation(features, key_output.representation), key_output
        )

    def predict(self, features: Tensor, initial_values: Tensor | None = None) -> Tensor:
        """Score candidates without future tree information.

        If the caller does not supply ``psi(v0)``, it is read from the first
        normalized bound feature, which is the current node objective divided
        by the root objective and is available during online search.
        """
        if initial_values is None:
            bound_index = sum(list(FEATURE_SLICES.values())[:10])
            psi_v0 = features[:, bound_index].clamp(0.0, 1.0)
            initial_values = torch.stack((torch.zeros_like(psi_v0), psi_v0), dim=1)
        key_output = self.key_sequence.generate(initial_values)
        return self.score_from_representation(features, key_output.representation)
