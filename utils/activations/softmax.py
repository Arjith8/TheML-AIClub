from typing import override

import torch

from nn.layers.layer import Layer

class Softmax(Layer):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.exp(x)
        denominator = x.sum()
        return x / denominator

    @override
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
