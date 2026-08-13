from typing import override

import torch

from nn.layers.layer import Layer

class Sigmoid(Layer):

    @property
    @override
    def parameters(self) -> list[torch.Tensor]:
        return []

    @override
    def forward(self, x: torch.Tensor):
        return 1/(1 + torch.exp(-x))

    @override
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
