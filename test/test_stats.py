import pytest

from src.stats.cossim_stats import MeanStats


@pytest.fixture()
def mean_stats():
    return MeanStats()


def test_update_moving_average(mean_stats):
    moving_averages = []

    values = [1, 2, 3, 4, 5]
    avg = 0
    for i in range(5):
        avg = mean_stats.update_moving_average(avg, values[i], i + 1)
        moving_averages.append(avg)

    expected_moving_averages = [1, 1.5, 2, 2.5, 3]
    assert expected_moving_averages == moving_averages
