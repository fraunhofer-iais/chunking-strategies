from src.config.config import DocumentEvaluatorConfig, EvaluatorConfig, QuestionEvaluatorConfig
from src.evaluator.document_evaluator import DocumentEvaluator
from src.evaluator.evaluator import Evaluator
from src.evaluator.question_evaluator import QuestionEvaluator


class EvaluatorFactory:
    @staticmethod
    def create(evaluator_config: EvaluatorConfig) -> Evaluator:
        if isinstance(evaluator_config, DocumentEvaluatorConfig):
            return DocumentEvaluator(evaluator_config=evaluator_config)
        elif isinstance(evaluator_config, QuestionEvaluatorConfig):
            return QuestionEvaluator(evaluator_config=evaluator_config)
