import torch

from examples.data_mocks.classification_tasks import create_classification_dataset
from nn.container.sequential import Sequential
from nn.layers.linear import Linear


def main():
    _ = torch.manual_seed(42)       # pyright: ignore[reportUnknownMemberType]
    num_classes = 4
    num_fields = 10
    data = create_classification_dataset(10, 1000, num_classes)

    model = Sequential([
        Linear(in_features=num_fields, out_features=num_classes),
        Linear(in_features=num_fields, out_features=num_classes),
    ])
    pass

if __name__ == "__main__":
    main()
