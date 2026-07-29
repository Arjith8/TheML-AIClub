from autograd.data_structures import Value


def main():
    a, b = Value(10), Value(11)
    c = a + 1
    print(c.val, c.parents, c.gradient)
    pass

if __name__ == "__main__":
    main()
