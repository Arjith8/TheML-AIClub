from typing import override

import torch

from nn.layers.layer import Layer

class Softmax(Layer):
    @property
    @override
    def parameters(self) -> list[torch.Tensor]:
        return []

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.max(dim=1, keepdim=True).values
        x = torch.exp(x)
        denominator = x.sum(dim=1, keepdim=True)
        return x / denominator

    @override
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
