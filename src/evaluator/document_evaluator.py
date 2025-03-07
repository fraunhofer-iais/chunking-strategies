from typing import List

from src.config.config import EvaluatorConfig
from src.dto.dto import DocumentEvalResult
from src.dto.dto import EvalSample, RetrieverResult
from src.evaluator.evaluator import Evaluator
from src.utils import mean_of_lists


class DocumentEvaluator(Evaluator):
    def evaluate(self, eval_sample: EvalSample, retrieved_paragraphs: List[RetrieverResult], k:int) -> DocumentEvalResult:
        """
        Evaluates the retrieval performance of the retriever model by comparing the predicted paragraphs
        """
        recalls_for_all_questions = []
        for question, answer, retrieved_paragraph in zip(eval_sample.questions, eval_sample.answers,
                                                         retrieved_paragraphs):
            recalls_for_all_questions.append(self.recall_at_k(answer.answer, k, retrieved_paragraph.paragraphs))
        mean_recalls_at_k = mean_of_lists(recalls_for_all_questions)
        result = DocumentEvalResult(
            recall_at_k=mean_recalls_at_k,
            eval_sample=eval_sample,
            retriever_results=retrieved_paragraphs,
        )
        return result
