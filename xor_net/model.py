import torch

from utils.activations.relu import RELU
from utils.activations.sigmoid import sigmoid
from utils.loss_functions.bce import BCE
from utils.plot import LossStep


class XORNet:
    def __init__(self, input_dim: int = 2, hidden_layer_neurons: int = 1) -> None:
        assert hidden_layer_neurons > 0
        self.w: torch.Tensor = torch.randn(input_dim, hidden_layer_neurons, requires_grad=True)
        self.b: torch.Tensor = torch.randn(1, hidden_layer_neurons, requires_grad=True)
        self.hidden_layer_neurons: int = hidden_layer_neurons

        self.out_w: torch.Tensor = torch.randn(hidden_layer_neurons, 1, requires_grad=True)
        self.out_b: torch.Tensor = torch.randn(1, requires_grad=True)
        self.flag = True

    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1000, log_step:int = 100, learning_rate: float = 0.01):
        loss_history: list[LossStep] = []
        for epoch in range(epochs):
            if self.flag:
                print("x", x)
                print("y", y)
                print("w", self.w)
                print("b", self.b)
            if epoch == 10:
                self.flag = False
            y_pred = self.forward(x)
            loss = self.error(y_pred, y)
            if not epoch % log_step:
                loss_history.append(LossStep(loss.item(), epoch))

            loss.backward()                         # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]
            with torch.no_grad():
                assert self.w.grad is not None
                assert self.b.grad is not None
                assert self.out_w.grad is not None
                assert self.out_b.grad is not None

                w_grad = self.w.grad
                b_grad = self.b.grad
                out_w_grad = self.out_w.grad
                out_b_grad = self.out_b.grad
                if self.flag:
                    print("y_pred: ", y_pred)
                    print("loss: ", loss)
                    print(f"""
                    w_grad: \t{w_grad}
                    b_grad: \t{b_grad}
                    out_w_grad:\t {out_w_grad}
                    out_b_grad:\t {out_b_grad}
                    """)
                    print()

                self.out_w -= learning_rate * out_w_grad
                self.out_b -= learning_rate * out_b_grad

                self.w -= learning_rate * w_grad
                self.b -= learning_rate * b_grad

                _ = w_grad.zero_()
                _ = b_grad.zero_()
                _ = out_w_grad.zero_()
                _ = out_b_grad.zero_()

        return loss_history
    
    def error(self, y_pred: torch.Tensor, y: torch.Tensor):
        return BCE(y_pred = y_pred, y = y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(x @ self.w + self.b)
        h = sigmoid(z @ self.out_w + self.out_b)
        
        if self.flag:
            print("z", z)
            print("h", h)
        return h
    def activation(self, z: torch.Tensor):
        return RELU(z)
