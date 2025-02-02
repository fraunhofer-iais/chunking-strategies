import unittest

from src.config.config import EvaluatorConfig
from src.dto.dto import EvalResult, EvalSample, RetrieverResult, Answer
from src.evaluator.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator_config = EvaluatorConfig()
        self.evaluator = Evaluator(self.evaluator_config)

    def test_evaluate(self):
        eval_sample = EvalSample(
            document_id="1",
            document="",
            questions=["What is AI?", "Define machine learning."],
            answers=[Answer(answer= "Artificial Intelligence"), Answer(answer= "Machine Learning")]
        )
        retrieved_paragraphs = [
            RetrieverResult(
                document_id="1",
                question = "What is AI?",
                scores = [1., 1., 1., 1., 1.],
                paragraphs=[
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
            ]),
            RetrieverResult(
                document_id="1",
                question = "Define machine learning.",
                scores = [1., 1., 1., 1., 1.],
                paragraphs=[
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
            ])
        ]

        result = self.evaluator.evaluate(eval_sample, retrieved_paragraphs)

        self.assertIsInstance(result, EvalResult)
        self.assertEqual(result.recall_at_k, [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(result.eval_sample, eval_sample)
        self.assertEqual(result.retriever_results, retrieved_paragraphs)

    def test_recall_at_k(self):
        answer = "Artificial Intelligence"
        k = 3
        retrieved_paragraphs = [
            "Artificial Intelligence is the simulation of human intelligence processes by machines.",
            "AI is a branch of computer science.",
            "It involves the development of algorithms."
        ]

        recalls = self.evaluator.recall_at_k(answer, k, retrieved_paragraphs)

        self.assertEqual(recalls, [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
