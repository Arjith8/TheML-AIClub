from typing import override

import torch

from utils.optimizer.base import Optimizer


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, params: list[torch.Tensor], beta: float) -> None:
        self.learning_rate: float = learning_rate
        self.params: list[torch.Tensor] = params
        self.u: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]
        self.beta: float = beta

    @override
    def step_zero_grad(self) -> None:
        for i, param in enumerate(self.params):
            prev_u = self.u[i]

            assert param.grad is not None
            grad = param.grad

            u = self.beta * prev_u + (1 - self.beta) * grad * grad

            eps = torch.finfo(u.dtype).eps
            adjusted_lr = self.learning_rate / (torch.sqrt(u + eps))
            param -= adjusted_lr * grad

            self.u[i] = u
            _ = grad.zero_()

