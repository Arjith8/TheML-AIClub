import torch

from utils.loss_functions.mse import MSE
from utils.plot import LossStep

class LinearRegression:
    def __init__(self, input_dim: int = 1) -> None:
        self.w: torch.Tensor = torch.randn(input_dim, 1, requires_grad=True)
        self.b: torch.Tensor = torch.randn(1, requires_grad=True)

    def error(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return MSE(y, y_pred)

    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1000, step:int = 100, learning_rate: float = 0.01):
        if epochs <= 0:
            raise ValueError("epoch must be positive")

        loss_history: list[LossStep] = []
        for epoch in range(epochs):
            y_pred = x @ self.w + self.b
            loss = self.error(y, y_pred)
            if not epoch % step:
                loss_history.append(LossStep(loss.item(), epoch))

            loss.backward()                         # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]
            with torch.no_grad():
                assert self.w.grad is not None
                assert self.b.grad is not None

                w_grad = self.w.grad
                b_grad = self.b.grad

                self.w -= learning_rate * w_grad
                self.b -= learning_rate * b_grad

                _ = w_grad.zero_()
                _ = b_grad.zero_()

        return loss_history
