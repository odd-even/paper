# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(arr, n, k):
    """
    :param np.array arr: Minority class sample, shape of [size, attributes]
    :param float|int n: Rate of SMOTE
    :param int k: Number of nearest neighbors
    :return np.array: re-sampling array data, shape of [size * int(n), attributes]
    """
    if n <= 0:
        raise Exception("Rate must be positive.")

    size, size_attr = arr.shape
    if n < 1:
        arr = arr[np.random.randint(size, size=round(size * n))]
        n = 1

    neigh = NearestNeighbors(k * 2)
    neigh.fit(arr)

    def get_neighbor_indices(elm):
        return neigh.kneighbors([elm], k, return_distance=False)[0]

    def populate():
        for _ in range(int(n)):
            for elm in arr:
                nei = arr[get_neighbor_indices(elm)[np.random.randint(k)]]
                yield elm + ((nei - elm) * np.random.rand(size_attr))

    return np.array(list(populate()))


if __name__ == '__main__':
    sample = np.random.rand(100 * 10).reshape((100, 10))
    oversampled = smote(sample, 2, 5)
    assert oversampled.shape == (200, 10)
    print(np.mean(sample, axis=0))
    print(np.mean(oversampled, axis=0))

