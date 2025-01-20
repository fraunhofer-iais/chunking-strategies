import json
import os
from typing import List

import numpy as np

from src.config.config import EvaluatorConfig
from src.dto.dto import RetrieverResult, RetrieverResults
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

        @param eval_sample: The ground truth EvalSample containing questions, answers, and document ID.
        @param predictions: List of lists of RetrievedParagraphs, each list corresponding to a question in the eval sample.

        @return: RetrieverResult object containing the evaluation metrics (MAP, MRR) and a detailed summary.
        """
        relevances = []
        sample_results = []

        for question, prediction in zip(eval_sample.questions, predictions):
            predicted_document_ids = [item.document_id for item in prediction]
            relevance = self.__get_doc_relevance(predicted_document_ids=predicted_document_ids,
                                                 expected_document_ids=[
                                                     eval_sample.document_id])  # currently one relevant doc per question

            relevances.append(relevance)

            sample_results.append(
                {
                    "question": question,
                    "expected_document_id": eval_sample.document_id,
                    "predicted_document_ids": predicted_document_ids,
                }
            )

        # calculates map and mrr over all questions in the document
        mean_reciprocal_rank = self.metrics.mean_reciprocal_rank(relevance_score=relevances)
        mean_average_precision = self.metrics.mean_reciprocal_rank(relevance_score=relevances)

        results = RetrieverResult(
            map=mean_average_precision,
            mrr=mean_reciprocal_rank,
            detailed_summary=sample_results,
            relevance_indicators=relevances
        )
        return results

    @staticmethod
    def __get_doc_relevance(predicted_document_ids: List[str], expected_document_ids: List[str]) -> List[bool]:
        """
        Compares the predicted document IDs against the expected document IDs.Compares the predicted document IDs against the expected document IDs.

                @param predictions:['doc1', 'doc2', 'doc3']
                @param expected_document_ids: ['doc1', 'doc2', 'doc5']
                @return: [[True, False, False], [True, True, False], [False, False, False]]
                """
        document_relevance = []

        # Iterate over predicted document IDs
        for predicted_document_id in predicted_document_ids:
            matches = []

            # Compare each predicted document ID with the ground truth document IDs
            for expected_document_id in expected_document_ids:
                is_match = (predicted_document_id == expected_document_id)  # Check if doc IDs match
                matches.append(is_match)

            document_relevance.append(any(matches))

        return document_relevance

    def evaluate_multiple_documents(self, eval_samples: List[EvalSample],
                                    predictions: List[List[List[RetrievedParagraphs]]]):
        """
         Evaluates retrieval metrics (MAP, MRR) over multiple documents in the evaluation set.
        @param eval_samples:
        @param predictions:
        @return:
        """
        mrr, map = list(), list()
        per_eval_sample_results = []

        # evaluate each document
        for eval_sample, prediction in zip(eval_samples, predictions):
            result_single_doc: RetrieverResult = self.evaluate_single_document(eval_sample, prediction)
            mrr.append(result_single_doc.mrr)
            map.append(result_single_doc.map)

            per_eval_sample_results.append(result_single_doc)

        # compute overall map and mrr for all documents
        overall_results = RetrieverResults(
            map_documents=float(np.average(map)),
            mrr_documents=float(np.average(mrr)),
            per_eval_sample_results=per_eval_sample_results
        )

        self.save_results_to_json(results=overall_results)

        return overall_results

    def save_results_to_json(self, results: RetrieverResults):
        results_dict = results.dict()

        os.makedirs(self.evaluator_config.output_dir, exist_ok=True)
        file_path = os.path.join(self.evaluator_config.output_dir,
                                 self.evaluator_config.output_file_name)

        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {file_path}")


if __name__ == "__main__":
    # Example retrieved data
    eval_sample = [
        EvalSample(
            document_id="doc1",
            document="This is a sample document. It contains important information about AI.",
            questions=["What is the first document about?", "What does first document contain?"],
            answers=[Answer(answer="AI"), Answer(answer="information")]
        ),
        EvalSample(document_id="doc2",
                   document="This is a sample document. It contains important information about AI.",
                   questions=["What is the second document about?", "What does second document contain?"],
                   answers=[Answer(answer="AI"), Answer(answer="information")])
    ]

    retrieved_paragraphs = [[[
        RetrievedParagraphs(
            document_id="doc1",
            question="What is the first document about?",
            paragraphs=["This is a sample document.", "It contains important information about AI."]
        ),
        RetrievedParagraphs(
            document_id="doc2",
            question="What is the first document about?",
            paragraphs=["This is a sample document.", "It contains important information about AI."]
        )
    ],
        [
            RetrievedParagraphs(
                document_id="doc2",
                question="What does first document contain?",
                paragraphs=["This is a sample document.", "It contains important information about AI."]
            ),
            RetrievedParagraphs(
                document_id="doc1",
                question="What does first document contain??",
                paragraphs=["This is a sample document.", "It contains important information about AI."]
            )
        ]
    ],
        [
            [
                RetrievedParagraphs(
                    document_id="doc2",
                    question="What is the second document about?",
                    paragraphs=["This is a sample document.", "It contains important information about AI."]
                ),
                RetrievedParagraphs(
                    document_id="doc1",
                    question="What is the second document about?",
                    paragraphs=["This is a sample document.", "It contains important information about AI."]
                )
            ],
            [
                RetrievedParagraphs(
                    document_id="doc2",
                    question="What does second document contain?",
                    paragraphs=["This is a sample document.", "It contains important information about AI."]
                ),
                RetrievedParagraphs(
                    document_id="doc1",
                    question="What does second document contain??",
                    paragraphs=["This is a sample document.", "It contains important information about AI."]
                )
            ]
        ]
    ]

    call_eval = Evaluator(evaluator_config=EvaluatorConfig())
    #evaluation_results = call_eval.evaluate_single_document(eval_sample, retrieved_paragraphs)
    evaluation_results = call_eval.evaluate_multiple_documents(eval_sample, retrieved_paragraphs)


