import torch

from linear_regression.plot import LossStep

class LinearRegression:
    def __init__(self) -> None:
        self.w: torch.Tensor = torch.randn(1, 1, requires_grad=True)
        self.b: torch.Tensor = torch.randn(1, requires_grad=True)

    def error(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return ((y-y_pred)**2).mean()

    def train(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 1000, step:int = 100, learning_rate: float = 0.01):
        if epoch <= 0:
            raise ValueError("epoch must be positive")

        start = epoch

        loss_history: list[LossStep] = []
        while epoch:
            lr = learning_rate
            y_pred = self.w * x + self.b
            loss = self.error(y, y_pred)
            if not epoch % step:
                loss_history.append(LossStep(loss.item(), start - epoch))

            loss.backward() # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]
            with torch.no_grad():
                assert self.w.grad is not None
                assert self.b.grad is not None

                w_grad = self.w.grad
                b_grad = self.b.grad

                self.w -= lr * w_grad
                self.b -= lr * b_grad

                _ = w_grad.zero_()
                _ = b_grad.zero_()

            epoch -= 1

        return loss_history

