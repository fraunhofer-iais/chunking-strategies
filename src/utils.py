import os
from datetime import datetime
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


def create_dir_if_not_exists(file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def current_datetime(fmt: str = "%m%d%Y_%H%M%S") -> str:
    now = datetime.now()
    date_time = now.strftime(fmt)
    return date_time
