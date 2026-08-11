import torch
from dataclasses import dataclass


@dataclass
class RegressionData:
    X: torch.Tensor
    y: torch.Tensor
    weights: torch.Tensor
    bias: torch.Tensor

def create_regression_dataset(
    num_params: int,
    dataset_size: int,
    noise: float = 0.1,
) -> RegressionData:
    X = torch.randn(dataset_size, num_params)

    weights = torch.randn(num_params, 1)
    bias = torch.randn(1)

    y = X @ weights + bias

    if noise > 0:
        y += torch.randn_like(y) * noise

    return RegressionData(
        X=X,
        y=y,
        weights=weights,
        bias=bias,
    )
