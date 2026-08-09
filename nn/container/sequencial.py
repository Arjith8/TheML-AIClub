import torch

from nn.layers.layer import Layer


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers

    def forward(self, data: torch.Tensor):
        activation = data
        for layer in self.layers:
            activation = layer(activation)
