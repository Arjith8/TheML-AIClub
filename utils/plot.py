from pathlib import Path
from typing import NamedTuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LossStep(NamedTuple):
    loss: float
    step: int

def loss_plot(data: list[LossStep], save_path: Path):
    df = pd.DataFrame(data)
    _ = sns.lineplot(data=df, x="step", y="loss")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss.png")    # pyright: ignore[reportUnknownMemberType]
    plt.show()                              # pyright: ignore[reportUnknownMemberType]
    plt.close()

def loss_plot_compare(
    data: dict[str, list[LossStep]],
    save_path: Path,
) -> None:
    rows: list[dict[str, float | str]] = []

    for optimizer, history in data.items():
        for step in history:
            rows.append(
                {
                    "optimizer": optimizer,
                    "step": step.step,
                    "loss": step.loss,
                }
            )

    df = pd.DataFrame(rows)

    _ = sns.lineplot(
        data=df,
        x="step",
        y="loss",
        hue="optimizer",
    )

    plt.tight_layout()
    plt.savefig(save_path / "loss_compare.png")   # pyright: ignore[reportUnknownMemberType]
    plt.show()                                    # pyright: ignore[reportUnknownMemberType]
    plt.close()
