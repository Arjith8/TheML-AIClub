# c7a9c53: Mmmm I am not as close to solution as I thought I will be I think I need a few more days, I have decided on the base
# data structure but the backwards method I need to think more about it I will need to know what the operators are too so ig thats the next step say but backwards how will i implement it....
from typing import Literal


type Operation = Literal["+", "*"] | None
class Value:
    def __init__(self, val: float, parents: list[Value] | None = None, operation: Operation = None):
        self.val: float = val
        self.parents: list[Value] = [] if parents is None else parents
        self.gradient: float = 0.0
        self.operation: Operation = operation

    def __add__(self, other: Value | int | float) -> Value:
        if not isinstance(other, Value):
            other = Value(other)

        return Value(val = self.val + other.val, parents=[self, other], operation="+")

    def __radd__(self, other: int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        return Value(val = self.val * other.val, parents=[self, other], operation="*")

    def __rmul__(self, other: int | float) -> Value:
        return self * other
    
    def backwards(self) -> None:
        if not self.gradient:
            self.gradient = 1.0
        

        parents = self.parents
        match self.operation:
            case "+":
                for parent in parents:
                    parent.gradient += self.gradient * 1
                    parent.backwards()
            case "*":
                parents[0].gradient += self.gradient * parents[1].val
                parents[1].gradient += self.gradient * parents[0].val
                parents[0].backwards()
                parents[1].backwards()
            case _:
                pass
