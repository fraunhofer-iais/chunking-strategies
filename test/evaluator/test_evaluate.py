import pytest
from src.dto.dto import EvalSample, Answer, RetrievedParagraphs, RetrieverResult, RetrieverResults
from src.config.config import EvaluatorConfig
from src.evaluator.evaluate import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator(EvaluatorConfig())


@pytest.fixture
def eval_samples():
    return [
        EvalSample(
            document_id="doc_001",
            document="In 2025, AI is transforming the way people interact with technology. AI applications "
                     "are becoming more pervasive in industries such as healthcare, finance, and education. "
                     "Many experts believe that the rapid development of AI will revolutionize jobs and productivity "
                     "in the next decade.",
            questions=["What are some industries affected by AI?", "What year is AI predicted to revolutionize jobs?"],
            answers=[
                Answer(answer="healthcare, finance, and education", start=133, end=182),
                Answer(answer="2025", start=0, end=4)
            ]
        ),
        EvalSample(
            document_id="doc_002",
            document="Climate change is a significant challenge facing the world today. It is caused by "
                     "human activities, such as burning fossil fuels, deforestation, and industrial processes. "
                     "Governments and organizations worldwide are working to mitigate its effects through policies "
                     "and sustainable practices.",
            questions=["What causes climate change?", "What is being done to combat climate change?"],
            answers=[
                Answer(answer="burning fossil fuels, deforestation, and industrial processes", start=62, end=118),
                Answer(answer="Governments and organizations are working to mitigate its effects", start=182, end=234)
            ]
        )
    ]


@pytest.fixture
def retrieved_paragraphs():
    return [
        [
            RetrievedParagraphs(
                document_id="doc_001",
                question="What are some industries affected by AI?",
                paragraphs=[
                    "In 2025, AI is transforming the way people interact with technology. AI applications are "
                    "becoming more pervasive in industries such as healthcare, finance, and education.",
                    "Many experts believe that the rapid development of AI will revolutionize jobs and "
                    "productivity in the next decade."
                ],
                scores=[0.95, 0.87]
            ),
            RetrievedParagraphs(
                document_id="doc_001",
                question="What year is AI predicted to revolutionize jobs?",
                paragraphs=[
                    "In 2025, AI is transforming the way people interact with technology. AI applications "
                    "are becoming more pervasive in industries such as healthcare, finance, and education.",
                    "Many experts believe that the rapid development of AI will revolutionize jobs and "
                    "productivity in the next decade."
                ],
                scores=[0.92, 0.85]
            )
        ],
        [
            RetrievedParagraphs(
                document_id="doc_002",
                question="What causes climate change?",
                paragraphs=[
                    "Climate change is a significant challenge facing the world today.",
                    "It is caused by human activities, such as burning fossil fuels, deforestation, and "
                    "industrial processes. Governments and organizations worldwide are working to mitigate "
                    "its effects through policies and sustainable practices."
                ],
                scores=[0.92, 0.88]
            ),
            RetrievedParagraphs(
                document_id="doc_002",
                question="What is being done to combat climate change?",
                paragraphs=[
                    "Climate change is a significant challenge facing the world today.",
                    "It is caused by human activities, such as burning fossil fuels, deforestation, "
                    "and industrial processes. Governments and organizations worldwide are working to "
                    "mitigate its effects through policies and sustainable practices."
                ],
                scores=[0.89, 0.83]
            )
        ]
    ]


def test_single_document_relevance(evaluator, eval_samples, retrieved_paragraphs):
    eval_sample = eval_samples[0]
    prediction = retrieved_paragraphs[0]

    result = evaluator.evaluate_single_document(eval_sample, prediction)

    assert isinstance(result, RetrieverResult)
    assert result.map >= 0
    assert result.mrr >= 0
    assert len(result.detailed_summary) == len(eval_sample.questions)

    question_result = result.detailed_summary[0]
    assert "question" in question_result
    assert "expected_answer" in question_result
    assert "predicted_chunks" in question_result
    assert question_result['expected_answer'] == eval_sample.answers[0]
    assert result.relevance_indicators[0] == [True, False]  # The first chunk should have relevance


def test_multiple_documents_relevance(evaluator, eval_samples, retrieved_paragraphs):
    eval_samples_list = eval_samples
    predictions = retrieved_paragraphs

    overall_results = evaluator.evaluate_multiple_documents(eval_samples_list, predictions)

    assert isinstance(overall_results, RetrieverResults)
    assert overall_results.map_documents >= 0
    assert overall_results.mrr_documents >= 0

    assert len(overall_results.per_eval_sample_results) == len(eval_samples_list)

    assert "question" in overall_results.per_eval_sample_results[0].detailed_summary[0]
    assert overall_results.per_eval_sample_results[0].relevance_indicators[0] == [True, False]


def test_chunk_relevance(evaluator):
    # Test when the answer is completely contained in the first chunk
    expected_answer = Answer(answer="healthcare, finance, and education", start=15, end=58)
    predicted_chunks = [
        "In 2025, AI is transforming the way people interact with technology. AI applications are becoming "
        "more pervasive in industries such as healthcare, finance, and education.",
        "Many experts believe that the rapid development of AI will revolutionize jobs and productivity "
        "in the next decade."
    ]
    relevance = evaluator.get_chunk_relevance(predicted_chunks, expected_answer)
    assert relevance == [True, False]

    # Test when the answer is contained in the second chunk
    expected_answer = Answer(answer="rapid development of AI", start=201, end=224)
    predicted_chunks = [
        "In 2025, AI is transforming the way people interact with technology. AI applications are "
        "becoming more pervasive in industries such as healthcare, finance, and education.",
        "Many experts believe that the rapid development of AI will revolutionize jobs "
        "and productivity in the next decade."
    ]
    relevance = evaluator.get_chunk_relevance(predicted_chunks, expected_answer)
    assert relevance == [False, True]

    # Test for case sensitivity
    expected_answer = Answer(answer="Rapid development of AI", start=201, end=224)
    predicted_chunks = [
        "In 2025, AI is transforming the way people interact with technology. AI applications are "
        "becoming more pervasive in industries such as healthcare, finance, and education.",
        "Many experts believe that the rapid development of AI will revolutionize jobs "
        "and productivity in the next decade."
    ]
    relevance = evaluator.get_chunk_relevance(predicted_chunks, expected_answer)
    assert relevance == [False, True]
