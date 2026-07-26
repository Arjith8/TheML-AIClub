import torch

from utils.activations.sigmoid import sigmoid

def create_dataset():
    true_w = torch.tensor([[3.0], [-2.0]])
    true_b = torch.tensor([1.0])

    x = torch.randn(1000, 2)

    z = x @ true_w + true_b
    z += 0.5 * torch.randn_like(z)

    p = sigmoid(z)
    y = (p >= 0.5).float()
    
    return (x, y)
