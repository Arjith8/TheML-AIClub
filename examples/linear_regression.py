from examples.data_mocks.regression_tasks import create_regression_dataset
from nn.container.sequencial import Sequential
from nn.layers.linear import Linear
from utils.loss_functions.mse import MSE
from utils.optimizer.momentum import Momentum


def main():
    data = create_regression_dataset(2, 1000)
    
    model = Sequential([
        Linear(2, 1)
    ])
    
    optimizer = Momentum(learning_rate=0.01, beta=0.9, params=model.parameters)
    epochs = 1000
    for _ in range(epochs):
        y_pred = model(data.X)
        loss = MSE(data.y, y_pred)
        loss.backward()                         # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]

        optimizer.step_zero_grad()

    print(model.parameters, data.weights, data.bias, sep="\n")

if __name__ == "__main__":
    main()
