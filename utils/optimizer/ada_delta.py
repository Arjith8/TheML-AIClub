from math import sqrt
from typing import override

import torch

from utils.optimizer.base import Optimizer


class AdaDelta(Optimizer):
    def __init__(self, params: list[torch.Tensor], beta: float) -> None:
        self.params: list[torch.Tensor] = params
        self.u: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]
        self.v: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]
        self.beta: float = beta

    @override
    def step_zero_grad(self) -> None:
        eps = torch.finfo(self.v[0].dtype).eps
        for i, param in enumerate(self.params):
            assert param.grad is not None
            grad = param.grad


            prev_v = self.v[i]
            v = self.beta * prev_v + (1 - self.beta) * torch.square(grad)

            prev_u = self.u[i]

            param_change = -(torch.sqrt(prev_u) + eps) / (torch.sqrt(v) + eps) * grad

            param += param_change

            u = self.beta * prev_u + (1 - self.beta) * torch.square(param_change)

            self.v[i] = v
            self.u[i] = u
            _ = grad.zero_()
