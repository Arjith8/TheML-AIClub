import torch


def sigmoid(z: torch.Tensor):
    return 1/(1 + torch.exp(-z))
