import torch


def cross_entropy(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return -(y * torch.log(y_pred.clamp(min=1e-7))).sum(dim=1, keepdim=True).mean()
