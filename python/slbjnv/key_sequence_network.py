"""Attention-based Seq2Seq network for paper-defined key-node value sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class KeySequenceOutput:
    predictions: Tensor
    mask: Tensor
    representation: Tensor
    attention: Tensor


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_states: Tensor, decoder_state: Tensor,
                mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        energy = self.energy(torch.tanh(
            self.encoder_projection(encoder_states)
            + self.decoder_projection(decoder_state).unsqueeze(1)
        )).squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, weights


class KeySequenceNetwork(nn.Module):
    """Encode ``{0, psi(v0)}`` and decode the remaining normalized values.

    Table 2 fixes the token width at three. A scalar sequence value is represented
    by its value, normalized position, and terminal-token flag.
    """

    def __init__(self, token_features: int = 3, embedding_size: int = 16,
                 hidden_size: int = 32, layers: int = 1, maximum_steps: int = 20,
                 end_tolerance: float = 0.02):
        super().__init__()
        if token_features != 3:
            raise ValueError("the paper-defined key token representation has width 3")
        self.maximum_steps = maximum_steps
        self.end_tolerance = end_tolerance
        self.embedding = nn.Linear(token_features, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, layers, batch_first=True)
        self.attention = AdditiveAttention(hidden_size)
        self.hidden_to_input = nn.Linear(hidden_size + embedding_size, embedding_size)
        self.decoder = nn.LSTM(embedding_size, hidden_size, layers, batch_first=True)
        self.hidden_to_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, 1)

    @staticmethod
    def _tokens(values: Tensor, positions: Tensor, terminal: Tensor) -> Tensor:
        return torch.stack((values, positions, terminal.to(values.dtype)), dim=-1)

    def _encode_initial(self, initial_values: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        if initial_values.ndim != 2 or initial_values.shape[1] != 2:
            raise ValueError("initial_values must have shape [batch, 2]")
        positions = torch.tensor((0.0, 1.0), device=initial_values.device,
                                 dtype=initial_values.dtype).expand_as(initial_values)
        terminal = torch.zeros_like(initial_values, dtype=torch.bool)
        embedded = self.embedding(self._tokens(initial_values, positions, terminal))
        states, hidden = self.encoder(embedded)
        mask = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        return states, hidden, mask

    def decode_step(self, previous_embedding: Tensor, hidden: tuple[Tensor, Tensor],
                    encoder_states: Tensor, mask: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        context, attention = self.attention(encoder_states, hidden[0][-1], mask)
        decoder_input = self.hidden_to_input(
            torch.cat((previous_embedding, context), dim=-1)
        ).unsqueeze(1)
        output, hidden = self.decoder(decoder_input, hidden)
        combined = torch.tanh(self.hidden_to_hidden(
            torch.cat((output[:, 0], context), dim=-1)
        ))
        prediction = self.hidden_to_output(combined).squeeze(-1)
        return prediction, hidden, attention

    def forward(self, value_sequences: Tensor, lengths: Tensor,
                teacher_forcing_ratio: float = 1.0) -> KeySequenceOutput:
        """Decode padded sequences that include the leading 0 and trailing 1."""
        if value_sequences.ndim != 2 or lengths.ndim != 1:
            raise ValueError("value_sequences and lengths must be rank 2 and rank 1")
        if value_sequences.shape[0] != lengths.shape[0] or torch.any(lengths < 3):
            raise ValueError("every sequence must contain start, at least one node, and end")
        encoder_states, hidden, encoder_mask = self._encode_initial(value_sequences[:, :2])
        step_count = int(lengths.max().item()) - 2
        previous_values = value_sequences[:, 1]
        previous_positions = torch.ones_like(previous_values)
        previous_terminal = torch.zeros_like(previous_values, dtype=torch.bool)
        predictions: list[Tensor] = []
        attentions: list[Tensor] = []
        for step in range(step_count):
            previous_embedding = self.embedding(self._tokens(
                previous_values, previous_positions, previous_terminal
            ))
            prediction, hidden, attention = self.decode_step(
                previous_embedding, hidden, encoder_states, encoder_mask
            )
            predictions.append(prediction)
            attentions.append(attention)
            target = value_sequences[:, step + 2]
            if teacher_forcing_ratio >= 1.0:
                previous_values = target
            elif teacher_forcing_ratio <= 0.0:
                previous_values = prediction
            else:
                use_teacher = torch.rand(prediction.shape, device=prediction.device) < teacher_forcing_ratio
                previous_values = torch.where(use_teacher, target, prediction)
            denominator = (lengths - 1).clamp_min(1).to(value_sequences.dtype)
            previous_positions = ((step + 2) / denominator).clamp_max(1.0)
            previous_terminal = (step + 2) == (lengths - 1)

        predicted = torch.stack(predictions, dim=1)
        attention_tensor = torch.stack(attentions, dim=1)
        positions = torch.arange(step_count, device=lengths.device).unsqueeze(0)
        mask = positions < (lengths - 2).unsqueeze(1)
        representation_mask = positions < (lengths - 3).clamp_min(0).unsqueeze(1)
        representation = value_sequences[:, 1] + (
            predicted * representation_mask.to(predicted.dtype)
        ).sum(dim=1)
        return KeySequenceOutput(predicted, mask, representation, attention_tensor)

    def generate(self, initial_values: Tensor, maximum_steps: int | None = None) -> KeySequenceOutput:
        """Autoregressively predict future key values for online scoring."""
        encoder_states, hidden, encoder_mask = self._encode_initial(initial_values)
        steps = maximum_steps or self.maximum_steps
        previous_values = initial_values[:, 1]
        active = torch.ones(initial_values.shape[0], dtype=torch.bool, device=initial_values.device)
        predictions: list[Tensor] = []
        masks: list[Tensor] = []
        attentions: list[Tensor] = []
        representation = initial_values[:, 1].clone()
        for step in range(steps):
            position = torch.full_like(previous_values, (step + 1) / max(steps, 1))
            previous_embedding = self.embedding(self._tokens(previous_values, position, ~active))
            prediction, hidden, attention = self.decode_step(
                previous_embedding, hidden, encoder_states, encoder_mask
            )
            prediction = prediction.clamp(0.0, 1.0)
            is_end = (prediction - 1.0).abs() <= self.end_tolerance
            include = active & ~is_end
            representation = representation + prediction * include.to(prediction.dtype)
            predictions.append(prediction)
            masks.append(active)
            attentions.append(attention)
            active = active & ~is_end
            previous_values = prediction
            if not bool(active.any()):
                break
        return KeySequenceOutput(
            torch.stack(predictions, dim=1),
            torch.stack(masks, dim=1),
            representation,
            torch.stack(attentions, dim=1),
        )
