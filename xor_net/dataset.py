import torch

def create_dataset(
    samples: int = 10000,
    input_dim: int = 2,
):
    x = torch.randint(0, 2, (samples, input_dim)).float()
    y = (x.sum(dim=1) % 2).float().unsqueeze(1)
    return x, y
