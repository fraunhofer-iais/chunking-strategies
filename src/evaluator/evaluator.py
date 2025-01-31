from typing import List

from src.config.config import EvaluatorConfig
from src.dto.dto import EvalResult
from src.dto.dto import EvalSample, RetrieverResult
from src.utils import create_list, mean_of_lists


class Evaluator:
    def __init__(self, evaluator_config: EvaluatorConfig):
        self.evaluator_config = evaluator_config

    def evaluate(self, eval_sample: EvalSample, retrieved_paragraphs: List[RetrieverResult]) -> EvalResult:
        """
        Evaluates the retrieval performance of the retriever model by comparing the predicted paragraphs
        """
        k = len(retrieved_paragraphs[0].paragraphs)
        recalls_for_all_questions = []
        for question, answer, retrieved_paragraph in zip(eval_sample.questions, eval_sample.answers,
                                                         retrieved_paragraphs):
            recalls_for_all_questions.append(self.recall_at_k(answer.answer, k, retrieved_paragraph.paragraphs))
        mean_recalls_at_k = mean_of_lists(recalls_for_all_questions)
        result = EvalResult(
            recall_at_k=mean_recalls_at_k,
            eval_sample=eval_sample,
            retriever_results=retrieved_paragraphs,
        )
        return result

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
