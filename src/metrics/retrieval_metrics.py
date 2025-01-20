import numpy as np
from typing import List


class RetrieverMetrics:
    """
    This is a class for calculating the retriever metrics_.
    """

    def __init__(self):
        pass

    def mean_reciprocal_rank(self, relevance_score: List[List[int]]) -> float:
        """
        :param relevance_score: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        :type relevance_score: List[List[int]]
        :return: Mean reciprocal rank (MRR). Score is reciprocal of the rank of the first relevant item
        First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
        Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
        0 - False
        1 - True
        relevance_score = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        mean_reciprocal_rank(relevance_score)
        0.61111111111111105
        relevance_score = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        mean_reciprocal_rank(relevance_score)
        0.5
        relevance_score = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
        mean_reciprocal_rank(relevance_score)
        0.75
        :rtype: float
        """
        relevance_score = (np.asarray(r).nonzero()[0] for r in relevance_score)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in relevance_score])

    def precision_at_k(self, relevance_score: List[bool], k: int) -> float:
        """
        :param relevance_score: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        :type relevance_score: List[bool]
        :return: Precision @ k.
        0 - False
        1 - True
        Relevance is binary (nonzero is relevant).
        relevance_score = [0, 0, 1]
        precision_at_k(relevance_score, 1)
        0.0
        precision_at_k(relevance_score, 2)
        0.0
        precision_at_k(relevance_score, 3)
        0.33333333333333331
        :rtype: float
        """
        assert k >= 1
        relevance_score = np.asarray(relevance_score)[:k] != 0
        if relevance_score.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(relevance_score)

    def average_precision(self, relevance_score: List[int]) -> float:
        """
        :param relevance_score: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        :type relevance_score: List[bool]
        :return: Average Precision. Score is average precision (area under PR curve)
        0 - False
        1 - True
        Relevance is binary (nonzero is relevant).
        relevance_score = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        average_precision(relevance_score)
         0.7833333333333333:rtype: float
        """
        relevance_score = np.asarray(relevance_score) != 0
        out = [self.precision_at_k(relevance_score, k + 1) for k in range(relevance_score.size) if relevance_score[k]]
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, relevance_score: List[List[int]]) -> float:
        """
        :param relevance_score: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        :type relevance_score: List[List[bool]]
        :return: Mean Average Precision (MAP). Score is mean average precision
        0 - False
        1 - True
        Relevance is binary (nonzero is relevant)
        relevance_score = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
        mean_average_precision(relevance_score)
        0.78333333333333333
        relevance_score = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
        mean_average_precision(relevance_score)
        0.39166666666666666
        :rtype: float
        """
        return np.mean([self.average_precision(r) for r in relevance_score])


if __name__ == '__main__':
    test_caller = RetrieverMetrics()

    # Example of relevance scores for two queries
    relevance_scores = [[1, 0, 0], [0, 0, 1]]

    print("MAP:   ", test_caller.mean_average_precision(relevance_scores))
    print("MRR:   ", test_caller.mean_reciprocal_rank(relevance_scores))

