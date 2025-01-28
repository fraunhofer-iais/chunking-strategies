from src.config.config import EvaluatorConfig
from src.evaluator.evaluate import Evaluator


class EvaluatorFactory:
    @staticmethod
    def create(evaluator_config: EvaluatorConfig) -> Evaluator:
        return Evaluator(evaluator_config=evaluator_config)
