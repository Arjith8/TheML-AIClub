import torch

# Mean Suqred Error btw

def MSE(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return ((y-y_pred)**2).mean()
