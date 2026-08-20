import torch


def shuffle(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rand_perm = torch.randperm(len(x))
    return x[rand_perm], y[rand_perm]
