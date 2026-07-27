import torch


def RELU(x: torch.Tensor):
    return x.clamp(min=0)
    
