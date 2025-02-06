from src.stats.cossim_stats import update_moving_average


def test_update_moving_average():        
    moving_averages = []

    values = [1, 2, 3, 4, 5]
    avg = 0
    for i in range(5):
        avg = update_moving_average(avg, values[i], i + 1)
        moving_averages.append(avg)

    expected_moving_averages = [1, 1.5, 2, 2.5, 3]
    assert expected_moving_averages == moving_averages
