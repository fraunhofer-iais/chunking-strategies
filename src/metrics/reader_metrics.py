import collections
import re
import string
from typing import List


class ReaderMetrics:
    """
    This is a class for calculating the reader metrics.
    """

    def __init__(self):
        pass

    def normalize_answer(self, sentence: str) -> str:
        """Preprocesses the sentence
        :param sentence: sentence string to be cleaned
        """

        def remove_articles(text: str) -> str:
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return str(text).lower()

        return white_space_fix(remove_articles(remove_punc(lower(sentence))))

    def get_tokens(self, sentence: str) -> List[str]:
        """Tokenises the sentence
        """
        if not sentence: return []
        return self.normalize_answer(sentence).split()

    def compute_exact(self, gold: str, pred: str) -> int:
        """
        EM - measures the proportion of cases where the predicted answer is identical to the correct answer
        :param gold: referernce answer
        :type gold: str
        :param pred: predicted answer
        :type pred: str
        :return Exact match metric. Returns 1 if the gold and prediction answers match exactly, 0 otherwise
        :rtype: int
        """
        return int(self.normalize_answer(gold) == self.normalize_answer(pred))

    def compute_f1_precision_recall(self, gold: str, pred: str) -> "tuple[float, float, float]":
        """
        :param gold: referernce answer
        :type gold: str
        :param pred: predicted answer
        :type pred: str
        :return: F1-Score,Precision and Recall based on common words
        :rtype: tuple[float, float, float]
        """
        gold_toks = self.get_tokens(gold)
        pred_toks = self.get_tokens(pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks), 0, 0
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall
