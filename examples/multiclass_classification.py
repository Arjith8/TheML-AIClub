import torch
from torch.nn.functional import one_hot

from examples.data_mocks.classification_tasks import create_classification_dataset
from nn.container.sequential import Sequential
from nn.layers.linear import Linear
from utils.activations.softmax import Softmax
from utils.loss_functions.ce import cross_entropy
from utils.optimizer.momentum import Momentum


def main():
    _ = torch.manual_seed(42)       # pyright: ignore[reportUnknownMemberType]
    num_classes = 4
    num_fields = 10
    data = create_classification_dataset(10, 1000, num_classes)
    y_ = one_hot(data.y.squeeze(), num_classes=num_classes).float()

    model = Sequential([
        Linear(in_features=num_fields, out_features=num_classes),
        Softmax()
    ])

    optimizer = Momentum(learning_rate=0.01, beta=0.9, params=model.parameters)
    epochs = 10000
    batch_size = 100
    data_size = data.X.size()[0]
    total_batch_iters = (data_size + batch_size - 1) // batch_size
    for epoch in range(epochs):
        rand_perm = torch.randperm(data_size)
        x = data.X[rand_perm]
        y = y_[rand_perm]
        for batch in range(total_batch_iters):
            batch_start_idx = batch*batch_size
            x_batch = x[batch_start_idx: batch_start_idx+batch_size]
            y_batch = y[batch_start_idx: batch_start_idx+batch_size]
            y_pred = model(x_batch)

            loss = cross_entropy(y_batch, y_pred)
            loss.backward()                         # pyright: ignore[reportUnusedCallResult, reportUnknownMemberType]

            optimizer.step_zero_grad()

        if epoch % 100 == 0:
            print(epoch, loss.item())

    print(model.parameters)
    print(model.forward(data.X))

if __name__ == "__main__":
    main()
