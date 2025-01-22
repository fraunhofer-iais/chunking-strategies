from typing import List

import numpy as np
from src.config.config import EvaluatorConfig
from src.dto.dto import RetrieverResult, RetrieverResults, Span
from src.dto.dto import EvalSample, RetrievedParagraphs, Answer
from src.metrics.retrieval_metrics import RetrieverMetrics


class Evaluator:
    def __init__(self, evaluator_config: EvaluatorConfig):
        self.evaluator_config = evaluator_config
        self.metrics = RetrieverMetrics()

    def evaluate_single_document(self, eval_sample: EvalSample, predictions: List[RetrievedParagraphs]) -> (
            RetrieverResult):
        """
        Evaluates retrieval metrics (MAP, MRR) for a single document by comparing predictions to ground truth.
        """
        relevances = []
        sample_results = []

        for question, answer, prediction in zip(eval_sample.questions, eval_sample.answers, predictions):
            predicted_chunks = prediction.paragraphs
            relevance = self.get_chunk_relevance(predicted_chunks=predicted_chunks, expected_answer=answer)

            relevances.append(relevance)  # Store relevance per question

            sample_results.append(
                {
                    "question": question,
                    "expected_answer": answer,
                    "predicted_chunks": predicted_chunks,
                }
            )

        # calculate MAP and MRR over all questions in the document
        mean_reciprocal_rank = self.metrics.mean_reciprocal_rank(relevance_score=relevances)
        mean_average_precision = self.metrics.mean_average_precision(relevance_score=relevances)

        results = RetrieverResult(
            map=mean_average_precision,
            mrr=mean_reciprocal_rank,
            detailed_summary=sample_results,
            relevance_indicators=relevances  # Save relevance indicators for each question
        )
        return results

    def get_chunk_relevance(self, predicted_chunks: List[str], expected_answer: Answer) -> List[
        bool]:
        """
        Given the predicted chunks (multiple paragraphs for each question) and expected answer,
        this function calculates the relevance of each chunk based on relative character indices.
        """
        relevance_scores = []
        start_idx = expected_answer.start  # Character-based start index of the expected answer
        answer_text = expected_answer.answer

        cumulative_offset = 0  # Track cumulative character offset across all chunks

        for chunk in predicted_chunks:
            chunk_start_idx = cumulative_offset  # This chunk starts at the cumulative offset
            chunk_end_idx = cumulative_offset + len(chunk)  # This chunk ends at cumulative offset + chunk length

            # Check if the expected answer's start index lies within the current chunk's range
            if chunk_start_idx <= start_idx < chunk_end_idx:
                # Answer is within this chunk, check if the chunk contains the answer text
                if answer_text in chunk:
                    relevance_scores.append(True)
                else:
                    relevance_scores.append(False)
            else:
                relevance_scores.append(False)

            cumulative_offset += len(chunk) + 1  # +1 to account for the space after each chunk

        return relevance_scores

    def evaluate_multiple_documents(self, eval_samples: List[EvalSample], predictions: List[List[RetrievedParagraphs]]) \
            -> RetrieverResults:
        """
        Evaluates retrieval metrics (MAP, MRR) over multiple documents in the evaluation set.
        """
        mrr, map = list(), list()
        per_eval_sample_results = []

        # evaluate each document
        for eval_sample, prediction in zip(eval_samples, predictions):
            result_single_doc: RetrieverResult = self.evaluate_single_document(eval_sample, prediction)
            mrr.append(result_single_doc.mrr)
            map.append(result_single_doc.map)

            per_eval_sample_results.append(result_single_doc)

        # compute overall MAP and MRR for all documents
        overall_results = RetrieverResults(
            map_documents=float(np.average(map)),
            mrr_documents=float(np.average(mrr)),
            per_eval_sample_results=per_eval_sample_results
        )
        return overall_results


if __name__ == "__main__":
    eval_samples = [
        EvalSample(
            document_id="doc_001",
            document="In 2025, AI is transforming the way people interact with technology. AI applications are becoming more pervasive in industries such as healthcare, finance, and education. Many experts believe that the rapid development of AI will revolutionize jobs and productivity in the next decade.",
            questions=["What are some industries affected by AI?", "What year is AI predicted to revolutionize jobs?"],
            answers=[
                Answer(answer="healthcare, finance, and education", start=133, end=182),
                Answer(answer="2025", start=0, end=4)
            ]
        ),
        EvalSample(
            document_id="doc_002",
            document="Climate change is a significant challenge facing the world today. It is caused by human activities, such as burning fossil fuels, deforestation, and industrial processes. Governments and organizations worldwide are working to mitigate its effects through policies and sustainable practices.",
            questions=["What causes climate change?", "What is being done to combat climate change?"],
            answers=[
                Answer(answer="burning fossil fuels, deforestation, and industrial processes", start=62, end=118),
                Answer(answer="Governments and organizations are working to mitigate its effects", start=182, end=234)
            ]
        )
    ]

    retrieved_paragraphs = [
        [
            RetrievedParagraphs(
                document_id="doc_001",
                question="What are some industries affected by AI?",
                paragraphs=[
                    "In 2025, AI is transforming the way people interact with technology. AI applications are becoming more pervasive in industries such as healthcare, finance, and education.",
                    "TMany experts believe that the rapid development of AI will revolutionize jobs and productivity in the next decade."
                ],
                scores=[0.95, 0.87]
            ),
            RetrievedParagraphs(
                document_id="doc_001",
                question="What year is AI predicted to revolutionize jobs?",
                paragraphs=[
                    "In 2025, AI is transforming the way people interact with technology. AI applications are becoming more pervasive in industries such as healthcare, finance, and education.",
                    "TMany experts believe that the rapid development of AI will revolutionize jobs and productivity in the next decade."
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
                    "It is caused by human activities, such as burning fossil fuels, deforestation, and industrial processes. Governments and organizations worldwide are working to mitigate its effects through policies and sustainable practices."
                ],
                scores=[0.92, 0.88]
            ),
            RetrievedParagraphs(
                document_id="doc_002",
                question="What is being done to combat climate change?",
                paragraphs=[
                    "Climate change is a significant challenge facing the world today.",
                    "It is caused by human activities, such as burning fossil fuels, deforestation, and industrial processes. Governments and organizations worldwide are working to mitigate its effects through policies and sustainable practices."

                ],
                scores=[0.89, 0.83]
            )
        ]
    ]

    evaluator = Evaluator(evaluator_config=EvaluatorConfig())
    evaluation_results = evaluator.evaluate_multiple_documents(eval_samples, retrieved_paragraphs)

    print("Evaluation Results:", evaluation_results)
