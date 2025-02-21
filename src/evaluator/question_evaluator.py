from typing import List

from src.dto.dto import QuestionEvalResult
from src.evaluator.document_evaluator import Evaluator


class QuestionEvaluator(Evaluator):
    def evaluate(self,document_id: str, question: str, answer: str, paragraphs: List[str], k:int) -> List[int]:
        """
        Evaluates the retrieval performance of the retriever model for a single question.
        """
        recall_at_k = self.recall_at_k(answer, k, paragraphs)

        eval_result = QuestionEvalResult(
            document_id=document_id,
            question=question,
            recall_at_k=recall_at_k,
        )
        return eval_result
