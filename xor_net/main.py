from pathlib import Path

import torch

from utils.optimizer.gradient_descent import GradientDescent
from utils.optimizer.momentum import Momentum
from xor_net.dataset import create_dataset
from xor_net.model import XORNet
from utils.plot import loss_plot

def main():
    _ = torch.manual_seed(42)       # pyright: ignore[reportUnknownMemberType]

    x, y = create_dataset()

    model = XORNet(hidden_layer_neurons=4)
    optimizer = Momentum(learning_rate = 0.1, params = model.parameters(), beta=0.9)

    loss_history = model.train(x, y, optimizer, epochs=10000, log_step = 1)

    save_path = Path(__file__).resolve().parent
    loss_plot(loss_history, save_path = save_path)

    print(f"Error History: {loss_history}")
    print(f"Weight: {model.w}")
    print(f"Bias:   {model.b}")

if __name__ == "__main__":
    main()
