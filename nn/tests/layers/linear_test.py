import torch

from nn.layers.linear import Linear


class TestLinear:
    def test_layer(self):
        layer = Linear(2, 1)
        layer.w.data = torch.tensor([[2.0], [3.0]])
        layer.b.data = torch.tensor([1.0])

        data = torch.tensor([[1, 2], [2, 3]], dtype=torch.float)

        output = layer(data)
        expected_output = torch.tensor([[9], [14]], dtype=torch.float)

        assert torch.equal(output, expected_output)
