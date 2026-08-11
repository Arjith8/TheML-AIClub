import torch

from nn.layers.layer import Layer


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        activation = data
        for layer in self.layers:
            activation = layer(activation)
        
        return activation

    @property
    def parameters(self) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters)

        return params

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
