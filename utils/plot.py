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
