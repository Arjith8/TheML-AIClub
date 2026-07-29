from typing import override

import torch

from utils.optimizer.base import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float, params: list[torch.Tensor]) -> None:
        self.learning_rate: float = learning_rate
        self.params: list[torch.Tensor] = params

    @override
    def step_zero_grad(self) -> None:
        for param in self.params:
            assert param.grad is not None
            grad = param.grad

            param -= self.learning_rate * grad
            _ = grad.zero_()
