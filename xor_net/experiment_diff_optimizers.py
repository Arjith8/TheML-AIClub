from pathlib import Path
from typing import Callable, TypeAlias

import torch

from utils.optimizer.ada_delta import AdaDelta
from utils.optimizer.ada_grad import AdaGrad
from utils.optimizer.adam import Adam
from utils.optimizer.base import Optimizer
from utils.optimizer.gradient_descent import GradientDescent
from utils.optimizer.momentum import Momentum
from utils.optimizer.rms_prop import RMSProp
from utils.plot import loss_plot_compare
from xor_net.dataset import create_dataset
from xor_net.model import XORNet
from utils.plot import LossStep

OptimizerFactory: TypeAlias = Callable[[XORNet], Optimizer]

def main() -> None:
    _ = torch.manual_seed(42)               # pyright: ignore[reportUnknownMemberType]
    x, y = create_dataset(input_dim=100)

    epochs = 50_000
    log_step = 10

    optimizers: list[tuple[str, OptimizerFactory]] = [
        (
            "Gradient Descent",
            lambda model: GradientDescent(
                learning_rate=0.1,
                params=model.parameters(),
            ),
        ),
        (
            "Momentum",
            lambda model: Momentum(
                learning_rate=0.1,
                beta=0.9,
                params=model.parameters(),
            ),
        ),
        (
            "AdaGrad",
            lambda model: AdaGrad(
                learning_rate=0.1,
                params=model.parameters(),
            ),
        ),
        (
            "AdaDelta",
            lambda model: AdaDelta(
                beta=0.95,
                params=model.parameters(),
            ),
        ),
        (
            "Adam",
            lambda model: Adam(
                learning_rate=0.001,
                m_beta=0.9,
                h_beta=0.999,
                params=model.parameters(),
            ),
        ),
        (
            "RMSProp",
            lambda model: RMSProp(
                learning_rate=0.001,
                beta=0.9,
                params=model.parameters(),
            ),
        ),
    ]

    histories: dict[str, list[LossStep]] = {}

    for name, optimizer_factory in optimizers:
        print("=" * 60)
        print(f"Training with {name}")

        _ = torch.manual_seed(42)                   # pyright: ignore[reportUnknownMemberType]

        model = XORNet(hidden_layer_neurons=100)
        optimizer = optimizer_factory(model)

        history = model.train(
            x,
            y,
            optimizer,
            epochs=epochs,
            log_step=log_step,
        )

        histories[name] = history

        print(f"Final Loss : {history[-1].loss:.6f}")
        print(f"Weight      : {model.w}")
        print(f"Bias        : {model.b}")

    save_path = Path(__file__).resolve().parent
    loss_plot_compare(histories, save_path=save_path)


if __name__ == "__main__":
    main()
