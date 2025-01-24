import time
from typing import List, Tuple, Union

import numpy as np
from src.config.config import EvaluatorConfig
from src.dto.dto import RetrieverResult, RetrieverResults, Span
from src.dto.dto import EvalSample, RetrievedParagraphs, Answer
from src.metrics.retrieval_metrics import RetrieverMetrics


class Evaluator:
    def __init__(self, evaluator_config: EvaluatorConfig):
        self.evaluator_config = evaluator_config
        self.metrics = RetrieverMetrics()

    def evaluate_single_document(self, eval_sample: EvalSample, predictions: List[RetrievedParagraphs]) -> tuple[
        RetrieverResult, int]:
        """
        Evaluates retrieval metrics (MAP, MRR) for a single document by comparing predictions to ground truth.
        """
        relevances = []
        sample_results = []

        for question, answer, prediction in zip(eval_sample.questions, eval_sample.answers, predictions):
            predicted_chunks = prediction.paragraphs
            relevance = self.get_chunk_relevance(predicted_chunks=predicted_chunks, expected_answer=answer)
            precision_all_k = self.metrics.precision_at_all_k(relevance_score=relevance, max_k=len(relevance))
            average_precision = self.metrics.average_precision(relevance_score=relevance)

            relevances.append(relevance)  # Store relevance per question

            sample_results.append(
                {
                    "question": question,
                    "expected_answer": answer,
                    "predicted_chunks": predicted_chunks,
                    "precision_at_all_k": precision_all_k,
                    "average_precision": average_precision,
                }
            )

        # calculate MAP and MRR over all questions in the document
        mean_reciprocal_rank = self.metrics.mean_reciprocal_rank(relevance_score=relevances)
        mean_average_precision = self.metrics.mean_average_precision(relevance_score=relevances)

        number_of_questions = len(eval_sample.questions)

        results = RetrieverResult(
            map=mean_average_precision,
            mrr=mean_reciprocal_rank,
            number_of_questions=number_of_questions,
            detailed_summary=sample_results,
            relevance_indicators=relevances  # Save relevance indicators for each question
        )
        return results, number_of_questions

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

        total_documents = len(eval_samples)
        total_questions = 0

        # evaluate each document
        for eval_sample, prediction in zip(eval_samples, predictions):
            result_single_doc, num_questions = self.evaluate_single_document(eval_sample, prediction)
            mrr.append(result_single_doc.mrr)
            map.append(result_single_doc.map)

            per_eval_sample_results.append(result_single_doc)
            total_questions += num_questions

        average_questions_per_document = total_questions / total_documents
        # compute overall MAP and MRR for all documents
        overall_results = RetrieverResults(
            map_documents=float(np.average(map)),
            mrr_documents=float(np.average(mrr)),
            per_eval_sample_results=per_eval_sample_results,
            total_documents=total_documents,
            total_questions=total_questions,
            average_questions_per_document=average_questions_per_document,

        )
        return overall_results
