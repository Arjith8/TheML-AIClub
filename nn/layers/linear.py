from typing import override

import torch

from nn.layers.layer import Layer

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.w: torch.Tensor = torch.rand(self.in_features, out_features, requires_grad=True)
        self.b: torch.Tensor = torch.rand(1, out_features, requires_grad=True)

    @override
    def forward(self, x: torch.Tensor):
        return x @ self.w + self.b

    @override
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
