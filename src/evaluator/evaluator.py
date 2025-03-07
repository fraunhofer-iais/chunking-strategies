from abc import ABC, abstractmethod
from typing import List

from src.config.config import EvaluatorConfig
from src.dto.dto import DocumentEvalResult
from src.dto.dto import EvalSample, RetrieverResult
from src.utils import create_list


class Evaluator(ABC):
    def __init__(self, evaluator_config: EvaluatorConfig):
        self.evaluator_config = evaluator_config

    @abstractmethod
    def evaluate(self, eval_sample: EvalSample, retrieved_paragraphs: List[RetrieverResult], k:int) -> DocumentEvalResult:
        pass

    def recall_at_k(self, answer: str, k: int, retrieved_paragraphs: List[str]) -> List[int]:
        recalls = []
        for idx, retrieved_paragraph in enumerate(retrieved_paragraphs):
            if answer.lower() in retrieved_paragraph.lower():
                # found the answer in the retrieved paragraphs
                recalls = create_list(k=k, idx=idx)
                break
        if recalls:
            return recalls
        else:
            return [0] * k