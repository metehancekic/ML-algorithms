from typing import TypeVar, Union, Tuple, Optional

import numpy as np
from numpy import array


class LinearRegression(object):
    r"""
    Linear regression algorithm
    """

    def __init__(self, learning_rate: float = 0.001, num_epochs: int = 100, batch_size: int = 100):
        r"""
        Args:
            learning_rate: Learning rate, default=0.001
            num_epochs: Number of epochs, default=100
            batch_size: Batch size, default=100
        """
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.weight = None
        self.bias = None

    def fit(self, data: array, targets: array, verbose: bool = False):
        r"""
        Args:
            data: Data
            targets: Target values
            verbose: Print loss at each epoch, default=False
        """
        num_samples, num_features = data.shape
        num_batches = num_samples // self.batch_size
        self.weight = np.random.randn(num_features)
        self.bias = 0

        for i in range(self.num_epochs):
            for batch_idx in range(num_batches):

                batch_data = data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                batch_targets = targets[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

                self.weight -= self.lr * (batch_data.T @ (self.predict(batch_data) - batch_targets) / num_samples)
                self.bias -= self.lr * np.sum(self.predict(batch_data) -
                                              batch_targets) / num_samples

            loss = self.mse(self.predict(data), targets)

            if verbose:
                print(f"Epoch {i} : loss {loss}")

    def least_squares(self):
        r"""
        TODO
        """
        pass

    def __call__(self, data: array) -> array:
        return self.predict(data)

    @staticmethod
    def mse(x: array, y: array) -> float:
        return np.mean((x - y)**2)

    def predict(self, data: array) -> array:
        return data @ self.weight + self.bias

    def __repr__(self) -> str:
        return f"LinearRegression(learning_rate={self.lr}, num_epochs={self.num_epochs}, batch_size={self.batch_size})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    points = [[0, 0], [1, 1], [1.9, 2], [3, 3.2], [4, 4.1], [5, 5.11]]
    X = np.array([[p[0]] for p in points])
    y = np.array([p[1] for p in points])

    regressor = LinearRegression(learning_rate=0.01, num_epochs=100, batch_size=2)
    regressor.fit(X, y, True)

    y_pred_line = regressor(X)

    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X, y, s=100)
    plt.plot(X, y_pred_line, color='black', linewidth=1, label="Prediction")
    plt.show()
