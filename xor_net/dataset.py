import torch

def create_dataset(input_dim: int = 2):
    assert input_dim > 1
    a: list[list[int]] = []
    b: list[int] = []

    for i in range(2**input_dim):
        temp: list[int] = []
        for j in range(input_dim):
            temp.append((i >> j) & 1)
        a.append(temp)

    for i in a:
        b.append(sum(i)%input_dim)


    x = torch.tensor(a, dtype=torch.float)
    y = torch.tensor(b, dtype=torch.float).unsqueeze(1)
    return x, y
