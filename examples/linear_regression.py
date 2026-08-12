import torch

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
    batch_size = 100
    data_size = data.X.size()[0]
    total_batch_iters = (data_size + batch_size - 1) // batch_size
    for epoch in range(epochs):
        rand_perm = torch.randperm(data_size)
        x = data.X[rand_perm]
        y = data.y[rand_perm]
        for batch in range(total_batch_iters):
            batch_start_idx = batch*batch_size
            x_batch = x[batch_start_idx: batch_start_idx+100]
            y_batch = y[batch_start_idx: batch_start_idx+100]
            y_pred = model(x_batch)
            loss = MSE(y_batch, y_pred)
            loss.backward()                         # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]
            if epoch % 100 == 0:
                print(epoch, loss.item())

            optimizer.step_zero_grad()

    print(model.parameters, data.weights, data.bias, sep="\n")

if __name__ == "__main__":
    main()
