import json
import os
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

    def evaluate_single_document(self, eval_sample: EvalSample, predictions: List[List[RetrievedParagraphs]]) -> (
            RetrieverResult):
        """
        Evaluates retrieval metrics (MAP, MRR) for a single document by comparing predictions to ground truth.
        """
        relevances = []
        sample_results = []

        for question, answer, prediction in zip(eval_sample.questions, eval_sample.answers, predictions):
            predicted_chunks = [prediction.paragraphs for prediction in prediction][0]
            relevance = self.__get_chunk_relevance(predicted_chunks=predicted_chunks, expected_answer=answer)

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

    def __get_chunk_relevance(self, predicted_chunks: List[str], expected_answer: Answer) -> List[bool]:
        """
        Given the predicted chunks (multiple paragraphs for each question) and expected answer,
        this function calculates the relevance of each chunk.
        """
        relevance_scores = []

        # Loop through each chunk for the question
        for chunk in predicted_chunks:
            relevance = self.__calculate_relevance_string_match(chunk, expected_answer)  # TODO: different relevance
            # scores can be explored
            relevance_scores.append(relevance)

        return relevance_scores

    def evaluate_multiple_documents(self, eval_samples: List[EvalSample], predictions: List[List[RetrievedParagraphs]]):
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

        self.save_results_to_json(results=overall_results)

        return overall_results

    def save_results_to_json(self, results: RetrieverResults):
        results_dict = results.model_dump()

        os.makedirs(self.evaluator_config.output_dir, exist_ok=True)
        file_path = os.path.join(self.evaluator_config.output_dir, self.evaluator_config.output_file_name)

        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {file_path}")

    @staticmethod
    def __calculate_relevance_string_match(chunk: str, expected_answer: Answer) -> bool:
        """
        Checks if the predicted chunk matches any part of the expected answer.
        Performs exact string matching (case-insensitive).
        """
        if expected_answer.answer.strip().lower() in chunk.strip().lower():
            return True
        else:
            return False


if __name__ == "__main__":
    # Example retrieved data
    eval_samples = [
        EvalSample(
            document_id="doc1",
            document="This document discusses the fundamentals of AI and its applications. The document highlights various AI technologies, including machine learning and deep learning.",
            questions=["What is this document about?", "What technologies are discussed in this document?"],
            answers=[Answer(answer="AI", start=0, end=2, spans=[Span(start=0, end=2)]),
                     Answer(answer="machine learning", start=0, end=2)]
        )
    ]

    retrieved_paragraphs = [[
        [
            RetrievedParagraphs(
                document_id="doc1",
                question="What is this document about?",
                paragraphs=["This document discusses the fundamentals of AI.",
                            "This document talks about AI in general.",
                            "AI and machine learning are discussed."],
                scores=[1, 1, 1]
            )
        ],
        [
            RetrievedParagraphs(
                document_id="doc1",
                question="What technologies are discussed in this document?",
                paragraphs=["This document talks about machine learning technologies.",
                            "AI technologies are discussed here.",
                            "Deep learning and neural networks are also mentioned."],
                scores=[1, 1, 1]
            )
        ]
    ]]

    evaluator = Evaluator(evaluator_config=EvaluatorConfig())
    evaluation_results = evaluator.evaluate_multiple_documents(eval_samples, retrieved_paragraphs)

    print("Evaluation Results:", evaluation_results)

