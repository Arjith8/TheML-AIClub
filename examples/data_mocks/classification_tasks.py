import torch
from dataclasses import dataclass


@dataclass
class ClassificationData:
    X: torch.Tensor
    y: torch.Tensor


def create_classification_dataset(
    num_params: int,
    dataset_size: int,
    num_classes: int,
) -> ClassificationData:
    X = torch.randn(dataset_size, num_params)

    class_weights = torch.randn(num_classes, num_params)

    scores = X @ class_weights.T
    y = torch.argmax(scores, dim=1)

    return ClassificationData(
        X=X,
        y=y,
    )
