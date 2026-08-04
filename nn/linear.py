import torch

class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.w: torch.Tensor = torch.rand(self.in_features, out_features, requires_grad=True)
        self.b: torch.Tensor = torch.rand(1, out_features, requires_grad=True)

