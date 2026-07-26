from pathlib import Path

from linear_regression.dataset import create_dataset
from linear_regression.model import LinearRegression
from utils.plot import loss_plot

def main():
    x, y = create_dataset()

    model = LinearRegression()

    loss_history = model.train(x, y, epochs=100, step = 1, learning_rate=0.1)

    save_path = Path(__file__).resolve().parent
    loss_plot(loss_history, save_path = save_path)

    print(loss_history)
    print(f"Weight: {model.w.item():.3f}")
    print(f"Bias:   {model.b.item():.3f}")

if __name__ == "__main__":
    main()
