import torch

class LinearRegression:
    def __init__(self) -> None:
        self.w: torch.Tensor = torch.randn(1, 1, requires_grad=True)
        self.b: torch.Tensor = torch.randn(1, requires_grad=True)

    def error(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return ((y-y_pred)**2).mean()

    def train(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 1000):
        if epoch <= 0:
            raise ValueError("epoch must be positive")

        loss = None
        while epoch:
            lr = 0.1
            y_pred = self.w * x + self.b
            loss = self.error(y, y_pred)
            if not epoch % 100:
                print(loss.item())

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

        return loss.detach() if loss else loss

