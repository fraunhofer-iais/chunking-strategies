from typing import List

import numpy as np


def create_list(k: int, idx: int):
    if idx > k:
        raise ValueError("Index cannot be greater than the length of the list.")
    return [0] * idx + [1] * (k - idx)


def mean_of_lists(lists: List[List[float]]) -> List[float]:
    # Transpose the list of lists
    transposed = np.transpose(lists)
    # Calculate the mean of each group
    means = [float(np.mean(group)) for group in transposed]
    return means
