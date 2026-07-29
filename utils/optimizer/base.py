from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step_zero_grad(self) -> None:
        raise NotImplementedError

