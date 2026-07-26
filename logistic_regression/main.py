from pathlib import Path

from logistic_regression.dataset import create_dataset
from logistic_regression.model import LogisticRegression
from utils.plot import loss_plot

def main():
    x, y = create_dataset()

    model = LogisticRegression(input_dim=2)

    loss_history = model.train(x, y, epochs=1000, log_step = 1, learning_rate=0.1)

    save_path = Path(__file__).resolve().parent
    loss_plot(loss_history, save_path = save_path)

    print(loss_history)
    print(f"Weight: {model.w}")
    print(f"Bias:   {model.b}")

if __name__ == "__main__":
    main()
