"""Three-hidden-layer score MLP from Table 2."""

from torch import Tensor, nn


class ScoreNetwork(nn.Module):
    def __init__(self, input_size: int = 17, hidden_size: int = 32):
        super().__init__()
        layers: list[nn.Module] = []
        size = input_size
        for _ in range(3):
            layers.extend((nn.Linear(size, hidden_size), nn.ReLU()))
            size = hidden_size
        layers.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs).squeeze(-1)
