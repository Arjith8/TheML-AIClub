import torch


def BCE(y: torch.Tensor, y_pred: torch.Tensor):
    return -1 * (y * torch.log(y_pred) + (1-y)*torch.log(1 - y_pred)).mean()
