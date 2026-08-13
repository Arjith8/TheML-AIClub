import torch


def BCE(y: torch.Tensor, y_pred: torch.Tensor):
    if y.shape != y_pred.shape:
        raise ValueError("Shapes of prediction and actual values dont match")
    eps = torch.finfo(y_pred.dtype).eps
    y_pred = y_pred.clamp(eps, 1 - eps)

    assert (y_pred < 1).all()
    assert (y_pred > 0).all()

    return -1 * (y * torch.log(y_pred) + (1-y)*torch.log(1 - y_pred)).mean()
