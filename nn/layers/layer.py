from abc import ABC

from torch import Tensor


class Layer(ABC):
    def forward(self, x: Tensor) -> Tensor:         # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError
    
    def __call__(self, data: Tensor) -> Tensor:
        raise NotImplementedError
