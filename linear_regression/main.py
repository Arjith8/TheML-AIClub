from linear_regression.dataset import create_dataset
from linear_regression.model import LinearRegression

def main():
    x, y = create_dataset()

    model = LinearRegression()

    final_loss = model.train(x, y)

    print(f"Weight: {model.w.item():.3f}")
    print(f"Bias:   {model.b.item():.3f}")
    print(f"Loss:   {final_loss:.3f}")

if __name__ == "__main__":
    main()
