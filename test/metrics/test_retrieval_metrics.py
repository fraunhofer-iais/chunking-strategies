from src.metrics.retrieval_metrics import RetrieverMetrics


def test_mean_reciprocal_rank():
    evaluate = RetrieverMetrics()
    relavance_score = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    expected_score = 0.61111111111111105
    predicted_score = evaluate.mean_reciprocal_rank(relavance_score)
    assert predicted_score == expected_score, 'Scores are same. Good!'


def test_precision_at_k():
    evaluate = RetrieverMetrics()
    relavance_score = [0, 0, 1]
    expected_score_at_1 = 0.0
    expected_score_at_2 = 0.0
    expected_score_at_3 = 0.33333333333333331
    predicted_score_at_1 = evaluate.precision_at_k(relavance_score, 1)
    predicted_score_at_2 = evaluate.precision_at_k(relavance_score, 2)
    predicted_score_at_3 = evaluate.precision_at_k(relavance_score, 3)
    assert predicted_score_at_1 == expected_score_at_1, 'Scores are same'
    assert predicted_score_at_2 == expected_score_at_2, 'Scores are same'
    assert predicted_score_at_3 == expected_score_at_3, 'Scores are same'


def test_average_precision():
    evaluate = RetrieverMetrics()
    relavance_score = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    expected_score = 0.7833333333333333
    predicted_score = evaluate.average_precision(relavance_score)
    assert predicted_score == expected_score, 'Score is same'


def test_mean_average_precision():
    evaluate = RetrieverMetrics()
    relavance_score = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    expected_score = 0.7833333333333333
    predicted_score = evaluate.mean_average_precision(relavance_score)
    assert predicted_score == expected_score, 'Score is same'

    relavance_score = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    expected_score = 0.39166666666666666
    predicted_score = evaluate.mean_average_precision(relavance_score)
    assert predicted_score == expected_score, 'Score is same'
