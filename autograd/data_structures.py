# c7a9c53: Mmmm I am not as close to solution as I thought I will be I think I need a few more days, I have decided on the base
# data structure but the backwards method I need to think more about it I will need to know what the operators are too so ig thats the next step say but backwards how will i implement it....
from typing import Literal


type Operation = Literal["+", "*"] | None
class Value:
    def __init__(self, val: float, parents: list[Value] | None = None, gradient: float = 0.0):
        self.val: float = val
        self.parents: list[Value] = [] if parents is None else parents
        self.gradient: float = gradient
        self.operation: Operation = None

    def __add__(self, other: Value | int | float) -> Value:
        print("Add")
        node = Value(self.val)
        node.parents.append(self)
        if not isinstance(other, Value):
            node.val += other
        else:
            node.val += other.val
            node.parents.append(other)
        node.operation = "+"
        return node

    def __radd__(self, other: int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        node = Value(val = self.val)
        node.parents.append(self)
        if not isinstance(other, Value):
            node.val *= other
        else:
            node.val *= other.val
            node.parents.append(other)
        node.operation = "*"
        return node

    def __rmul__(self, other: int | float) -> Value:
        return self * other
    
    def backwards(self) -> None:
        if not self.parents:
            self.gradient = 1.0
            return
        
        match self.operation:
            case "+":
                self.parents[0].backwards()
                self.gradient += self.parents[0].gradient
                if len(self.parents) == 2:
                    self.parents[1].backwards()
                    self.gradient += self.parents[1].gradient
            case _:
                print("unknown opn")
