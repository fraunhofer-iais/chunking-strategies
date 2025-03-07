import pytest

from src.config.config import EvaluatorConfig
from src.dto.dto import DocumentEvalResult, EvalSample, RetrieverResult, Answer
from src.evaluator.evaluator import Evaluator


@pytest.fixture
def evaluator():
    evaluator_config = EvaluatorConfig()
    evaluator = Evaluator(evaluator_config)
    return evaluator


def test_evaluate(evaluator):
    eval_sample = EvalSample(
        document_id="1",
        document="",
        questions=["What is AI?", "Define machine learning."],
        answers=[Answer(answer="Artificial Intelligence"), Answer(answer="Machine Learning")]
    )
    retrieved_paragraphs = [
        RetrieverResult(
            document_id="1",
            question="What is AI?",
            scores=[1., 1., 1., 1., 1.],
            paragraphs=[
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
                "Artificial Intelligence is the simulation of human intelligence processes by machines.",
            ]),
        RetrieverResult(
            document_id="1",
            question="Define machine learning.",
            scores=[1., 1., 1., 1., 1.],
            paragraphs=[
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
                "Machine Learning is a subset of AI that involves the use of algorithms and statistical models.",
            ])
    ]

    result = evaluator.evaluate(eval_sample=eval_sample, retrieved_paragraphs=retrieved_paragraphs, k=5)

    assert isinstance(result, DocumentEvalResult)
    assert result.recall_at_k == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert result.eval_sample == eval_sample
    assert result.retriever_results == retrieved_paragraphs


def test_recall_at_k(evaluator):
    answer = "Artificial Intelligence"
    k = 3
    retrieved_paragraphs = [
        "Artificial Intelligence is the simulation of human intelligence processes by machines.",
        "AI is a branch of computer science.",
        "It involves the development of algorithms."
    ]

    recalls = evaluator.recall_at_k(answer, k, retrieved_paragraphs)
    assert recalls == [1, 1, 1]
