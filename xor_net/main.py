from pathlib import Path

import torch

from xor_net.dataset import create_dataset
from xor_net.model import XORNet
from utils.plot import loss_plot

def main():
    _ = torch.manual_seed(42)       # pyright: ignore[reportUnknownMemberType]

    x, y = create_dataset()

    model = XORNet(hidden_layer_neurons=4)

    loss_history = model.train(x, y, epochs=10000, log_step = 1, learning_rate=0.01)

    save_path = Path(__file__).resolve().parent
    loss_plot(loss_history, save_path = save_path)

    print(f"Error History: {loss_history}")
    print(f"Weight: {model.w}")
    print(f"Bias:   {model.b}")

if __name__ == "__main__":
    main()
