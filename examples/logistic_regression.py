import torch

from examples.data_mocks.classification_tasks import create_classification_dataset
from nn.container.sequencial import Sequential
from nn.layers.linear import Linear
from nn.layers.sigmoid import Sigmoid
from utils.loss_functions.bce import BCE
from utils.optimizer.momentum import Momentum


def main():
    _ = torch.manual_seed(42)       # pyright: ignore[reportUnknownMemberType]
    num_classes = 2
    data = create_classification_dataset(2, 10, num_classes)
    
    model = Sequential([
        Linear(2, 1),
        Sigmoid(),
    ])
    
    optimizer = Momentum(learning_rate=0.01, beta=0.9, params=model.parameters)
    epochs = 10000
    batch_size = 100
    data_size = data.X.size()[0]
    total_batch_iters = (data_size + batch_size - 1) // batch_size
    for epoch in range(epochs):
        rand_perm = torch.randperm(data_size)
        x = data.X[rand_perm]
        y = data.y[rand_perm]
        for batch in range(total_batch_iters):
            batch_start_idx = batch*batch_size
            x_batch = x[batch_start_idx: batch_start_idx+batch_size]
            y_batch = y[batch_start_idx: batch_start_idx+batch_size]
            y_pred = model(x_batch).squeeze(1)

            loss = BCE(y_batch, y_pred)
            loss.backward()                         # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]
            if epoch % 100 == 0:
                print(epoch, loss.item())

            optimizer.step_zero_grad()

    print(model.parameters)

if __name__ == "__main__":
    main()
