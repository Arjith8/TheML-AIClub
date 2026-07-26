import torch
from utils.activations.sigmoid import sigmoid
from utils.loss_functions.bce import BCE
from utils.plot import LossStep


class LogisticRegression:
    def __init__(self, input_dim:int = 1) -> None:
        self.w: torch.Tensor = torch.randn(input_dim, 1, requires_grad=True)
        self.b: torch.Tensor = torch.randn(1, requires_grad=True)

    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1000, log_step:int = 100, learning_rate: float = 0.01):
        loss_history: list[LossStep] = []
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.loss(y_pred = y_pred, y = y)
            if not epoch % log_step:
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

    def forward(self, x: torch.Tensor):
        z = x @ self.w + self.b
        return sigmoid(z)
    
    def loss(self, y: torch.Tensor, y_pred: torch.Tensor):
        return BCE(y, y_pred)
