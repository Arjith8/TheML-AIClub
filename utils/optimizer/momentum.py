from typing import override

import torch

from utils.optimizer.base import Optimizer


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, beta: float, params: list[torch.Tensor]) -> None:
        self.learning_rate: float = learning_rate
        self.beta: float = beta
        self.params: list[torch.Tensor] = params
        self.u: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]

    @override
    def step_zero_grad(self) -> None:
        with torch.no_grad():
            for i, param in enumerate(self.params):
                prev_u = self.u[i]

                assert param.grad is not None
                grad = param.grad

                u = self.beta * prev_u + grad

                param -= self.learning_rate * u

                self.u[i] = u
                _ = grad.zero_()

