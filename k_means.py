from typing import TypeVar, Union, Tuple, Optional

import numpy as np
from numpy import array
import math


class KMeans(object):
    r"""
    K means clustering algorithm
    """

    def __init__(self, k: int, threshold: float = 0.00000001, max_iter: int = 1000):
        r"""
        Args:
            k: the number of clusters
            threshold: threshold between the loss values 
                               of the successive iterations to stop the algorithm,
                               default=0.00000001
            max_iter: Maximum number of iterations, default=1000
        """
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter

    def __call__(self, data: array, verbose: bool = False) -> Tuple[array, array, float]:
        r"""
        Args:
            data: data to be clustered
            verbose: Print running Loss, default=False
        """

        L = len(data)
        if self.k > L:
            raise ValueError("k cannot be larger than the size of data")

        means_indices = np.random.choice(L, size=self.k, replace=True)
        self.means = data[means_indices]
        predictions = np.zeros(L)
        class_distances = np.zeros((L, self.k))

        old_loss = math.inf
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            running_loss = 0
            for data_idx, datum in enumerate(data):

                class_distances[data_idx, :] = np.array(
                    [self.euclid_distance(datum, self.means[cls_idx]) for cls_idx in range(self.k)])

            # compute current loss
            running_loss = np.sum(np.min(class_distances, axis=1))

            # Check if it is improved
            if np.abs(old_loss - running_loss) < self.threshold:
                break

            # cluster according to distances
            self.predictions = np.argmin(class_distances, axis=1)

            # Compute new means
            self.means = np.array([np.mean(data[self.predictions == cls_idx, :], axis=0)
                                   for cls_idx in range(self.k)])

            # Set current loss as old loss
            old_loss = running_loss
            if verbose:
                print(f"Iteration: {iteration}, Loss: {old_loss/L}")

        return self.predictions, self.means, running_loss

    @staticmethod
    def euclid_distance(x: array, y: array) -> float:
        return np.sqrt(np.sum((x-y)**2))

    def __repr__(self) -> str:
        return f"KMeans(k={self.k}, threshold={self.threshold}, max_iter={self.max_iter})"


def gaussian_data(n_data: int = 1000, cluster_means: array = np.array([[0, 0], [10, 10]]), shuffle: bool = True) -> Tuple[array, array]:
    r"""
    Args:
        n_data: Number of datapoints for each cluster.
        cluster_means: Mean values for each cluster.
        shuffle: Shuffle the dataset.
    """

    # Initialize dataset shape(Number of clusters, n_data, data-point size)
    data = np.zeros((cluster_means.shape[0], n_data, cluster_means.shape[1]))

    # Initialize labels shape(Number of clusters, n_data)
    labels = cluster_means.shape[0]*[None]

    # Fill datapoints
    for k in range(cluster_means.shape[0]):
        data[k, :, :] = np.random.randn(n_data, cluster_means.shape[1]) + cluster_means[k]
        labels[k] = k * np.ones(n_data)

    data = data.reshape([-1, data.shape[-1]])
    labels = np.concatenate(tuple(labels))

    # Shuffle indexes
    data_idx = np.arange(len(data))
    np.random.shuffle(data_idx)

    # Shuffle data with shuffled indexes
    data = data[data_idx]
    labels = labels[data_idx]

    return data, labels


def main():
    # Create dataset
    data, labels = gaussian_data(n_data=1000, cluster_means=np.array([[0, 0], [10, 10]]))

    # Run k-means algorithm
    kmeans = KMeans(k=2)
    predictions, means, running_loss = kmeans(data, verbose=True)
    print(np.mean(predictions == labels))


if __name__ == "__main__":
    main()
