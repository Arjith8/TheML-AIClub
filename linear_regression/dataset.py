import torch

def create_dataset():
    x = torch.linspace(-5, 5, 100)
    noise = torch.rand_like(x) * .5
    
    y = 3*x + 5 + noise
    
    return (x, y)
