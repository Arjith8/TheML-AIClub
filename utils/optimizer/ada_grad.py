from typing import override

import torch

from utils.optimizer.base import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float, params: list[torch.Tensor]) -> None:
        self.learning_rate: float = learning_rate
        self.params: list[torch.Tensor] = params
        self.u: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]

    @override
    def step_zero_grad(self) -> None:
        for i, param in enumerate(self.params):
            prev_u = self.u[i]

            assert param.grad is not None
            grad = param.grad

            u = prev_u + grad * grad

            eps = torch.finfo(u.dtype).eps
            adjusted_lr = self.learning_rate / (torch.sqrt(u + eps))
            param -= adjusted_lr * grad

            self.u[i] = u
            _ = grad.zero_()

