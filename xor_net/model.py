import torch

from utils.activations.relu import RELU
from utils.activations.sigmoid import sigmoid
from utils.loss_functions.bce import BCE
from utils.plot import LossStep


class XORNet:
    def __init__(self,  input_dim: int = 2, hidden_layer_neurons: int = 1) -> None:
        assert hidden_layer_neurons > 0
        self.w: torch.Tensor = torch.randn(input_dim, hidden_layer_neurons, requires_grad=True)
        self.b: torch.Tensor = torch.randn(1, hidden_layer_neurons, requires_grad=True)
        self.hidden_layer_neurons: int = hidden_layer_neurons

        self.out_w: torch.Tensor = torch.randn(hidden_layer_neurons, 1, requires_grad=True)
        self.out_b: torch.Tensor = torch.randn(1, requires_grad=True)
        self.flag: bool = True

    def parameters(self):
        return [self.w, self.b, self.out_w, self.out_b]

    def train(self, x: torch.Tensor, y: torch.Tensor, optimizer, epochs: int = 1000, log_step:int = 100):
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
                optimizer.step_zero_grad()
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
