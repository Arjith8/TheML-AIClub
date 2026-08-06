# c7a9c53: Mmmm I am not as close to solution as I thought I will be I think I need a few more days, I have decided on the base
# data structure but the backwards method I need to think more about it I will need to know what the operators are too so ig thats the next step say but backwards how will i implement it....
from typing import Literal


type Operation = Literal["+", "*", "**"] | None
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

    def __sub__(self, other: Value | int | float) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        other = -1 * other
        return Value(val = self.val + other.val, parents=[self, other], operation="+")

    def __rsub__(self, other: int | float) -> Value:
        _other = Value(other)
        return _other - self

    def __pow__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            raise NotImplementedError("Value ** Value not implemented btw, needs quite a bit of thought")
        return Value(val = self.val ** other, parents=[self, Value(other)], operation='**')

    def __truediv__(self, other: Value | int | float) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        
        other = other ** -1
        return self * other

    def __rtruediv__(self, other: int | float) -> Value:
        return Value(other) / self

    def backwards(self):
        self.gradient = 1.0

        topologically_sorted_graph: list[Value] = []
        stack: list[tuple[Value, bool]] = []
        visited: set[Value] = set()
        visited.add(self)
        stack.append((self, False))
        while stack:
            current, expanded = stack.pop()
            if not expanded:
                stack.append((current, True))
                for i in current.parents:
                    if i not in visited:
                        visited.add(i)
                        stack.append((i, False))
                continue
            topologically_sorted_graph.append(current)
        for current in reversed(topologically_sorted_graph):
            parents = current.parents
            match current.operation:
                case "+":
                    parents[0].gradient += current.gradient * 1
                    parents[1].gradient += current.gradient * 1

                case "*":
                    parents[0].gradient += current.gradient * parents[1].val
                    parents[1].gradient += current.gradient * parents[0].val

                case "**":
                    parents[0].gradient = current.gradient * (parents[1].val) * parents[0].val ** (parents[1].val - 1)
                case _:
                    pass
