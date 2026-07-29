class Value:
    def __init__(self, val: float, parents: list[Value] | None = None, gradient: float = 0.0):
        self.val: float = val
        self.parents: list[Value] = [] if parents is None else parents
        self.gradient: float = gradient

    def __add__(self, other: Value | int | float) -> Value:
        node = Value(self.val)
        node.parents.append(self)
        if not isinstance(other, Value):
            node.val += other
        else:
            node.val += other.val
            node.parents.append(other)
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
        return node

    def __rmul__(self, other: int | float) -> Value:
        return self * other
