import json
import os

from src.config.config import JsonReaderConfig, QuestionEvaluatorConfig
from src.dto.dto import AverageDocResult, QuestionEvalResult
from src.evaluator.question_evaluator import QuestionEvaluator
from src.json_reader.json_reader import JsonReader
from src.utils import mean_of_lists

def post_evaluate_questions():
    """
    Reads all JSON files with DocumentEvalResults.
    Parses those results and computes the average recall at k for all questions.
    """
    print(os.getcwd())
    evaluator_config = QuestionEvaluatorConfig()
    question_evaluator = QuestionEvaluator(evaluator_config)
    
    json_reader = JsonReader(config=JsonReaderConfig())
    json_data = json_reader.read_jsons_from_directory()

    for file_path, exp_data in json_data.items():
        question_eval_results = []
        for doc in exp_data:
            document_id = doc["eval_sample"]["document_id"]
            k = len(doc["recall_at_k"])
            for retriever_result, answer_data in zip(doc["retriever_results"], doc["eval_sample"]["answers"]):
                question = retriever_result["question"]
                answer = answer_data["answer"]
                paragraphs = retriever_result["paragraphs"]

                question_eval_result = question_evaluator.evaluate(
                    document_id=document_id,
                    question=question,
                    answer=answer,
                    paragraphs=paragraphs,
                    k=k,
                )
                
                question_eval_results.append(question_eval_result)

        write_json(filepath=file_path.replace(".json", "_question_eval.json"), data=[result.model_dump() for result in question_eval_results])
        average_recall = _compute_average_recall(question_eval_results)
        write_json(filepath=file_path.replace(".json", "_question_average_scores.json"), data=average_recall.model_dump())


def _compute_average_recall(eval_results: list[QuestionEvalResult]) -> AverageDocResult:
    """Is used both for DocumentEvalResult and QuestionEvalResult."""
    list_of_recalls = [eval_result.recall_at_k for eval_result in eval_results]
    result = AverageDocResult(
        average_recall_at_k=mean_of_lists(list_of_recalls),
    )
    return result
        

def write_json(filepath: str, data: dict):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    post_evaluate_questions()
