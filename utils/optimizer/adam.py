from typing import override

import torch

from utils.optimizer.base import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate: float, params: list[torch.Tensor], h_beta: float, m_beta: float) -> None:
        self.learning_rate: float = learning_rate
        self.params: list[torch.Tensor] = params
        self.momentum: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]
        self.history: list[torch.Tensor] = [torch.zeros_like(i) for i in self.params]
        self.m_beta: float = m_beta
        self.h_beta: float = h_beta
        self.t: int = 0

    @override
    def step_zero_grad(self) -> None:

        self.t += 1
        for i, param in enumerate(self.params):
            assert param.grad is not None
            grad = param.grad

            momentum = self.m_beta * self.momentum[i] + (1 - self.m_beta) * grad
            momentum_bias_corrected = momentum / (1 - self.m_beta ** self.t)

            history = self.h_beta * self.history[i] + (1 - self.h_beta) * torch.square(grad)
            history_bias_corrected = history / (1 - self.h_beta ** self.t)

            eps = torch.finfo(history.dtype).eps

            adjusted_lr = self.learning_rate / (torch.sqrt(history_bias_corrected) + eps)
            param -= adjusted_lr * momentum_bias_corrected

            self.momentum[i] = momentum
            self.history[i] = history
            _ = grad.zero_()
