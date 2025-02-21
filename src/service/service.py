import argparse
import json
from typing import List

from llama_index.core import Document
from tqdm import tqdm

from src.config.config import TokenSplitterConfig, DocumentEvaluatorConfig, QuestionEvaluatorConfig,  EmbedModelConfig, VectorDBConfig, \
    DataHandlerConfig, SplitterConfig
from src.dto.dto import RetrieverResult, EvalSample, EvalResult, AverageDocResult
from src.factory.data_handler_config_factory import DataHandlerConfigFactory
from src.factory.data_handler_factory import DataHandlerFactory
from src.factory.embed_model_factory import EmbedModelFactory
from src.factory.evaluator_factory import EvaluatorFactory
from src.factory.splitter_factory import SplitterFactory
from src.utils import mean_of_lists, create_dir_if_not_exists, current_datetime
from src.vector_db.vector_db import VectorDB


class Service:
    def __init__(self,
                 splitter_configs: List[SplitterConfig],
                 document_evaluator_config: DocumentEvaluatorConfig,
                 question_evaluator_config: QuestionEvaluatorConfig,
                 embed_model_config: EmbedModelConfig,
                 data_handler_config: DataHandlerConfig,
                 vector_db_config: VectorDBConfig,
                 ):

        # to save all configs to json
        self.configs = [
            document_evaluator_config,
            question_evaluator_config,
            embed_model_config,
            data_handler_config,
            vector_db_config,
        ]
        print(f"Running with configs: {self.configs}")
        
        # objects are instantiated later
        self.vector_db_config = vector_db_config
        self.splitter_configs = splitter_configs

        # objects are instantiated here
        self.document_evaluator = EvaluatorFactory.create(evaluator_config=document_evaluator_config)
        self.question_evaluator = EvaluatorFactory.create(evaluator_config=question_evaluator_config)
        self.data_handler = DataHandlerFactory.create(data_handler_config=data_handler_config)
        # data is loaded here since it is then used for different chunk sizes
        self.data: List[EvalSample] = self.data_handler.load_data(limit=self.evaluator.evaluator_config.eval_limit)

        self.embed_model = EmbedModelFactory.create(embed_model_config=embed_model_config)

    def run(self):
        for splitter_config in self.splitter_configs:
            self.run_with_one_text_splitter(splitter_config=splitter_config)

    def run_with_one_text_splitter(self, splitter_config: SplitterConfig) -> List[DocumentEvalResult]:
        text_splitter = SplitterFactory.create(splitter_config=splitter_config)
        document_eval_results = []
        question_eval_results = []
        directory = self._construct_dir(text_splitter=text_splitter)
        self._save_all_configs(filename=directory.replace(".json", "_configs.json"), splitter_config=splitter_config)
        for sample in tqdm(
                self.data[self.evaluator.evaluator_config.eval_start:self.evaluator.evaluator_config.eval_limit]):
            doc = Document(text=sample.document, doc_id=sample.document_id)
            vector_db = VectorDB(documents=[doc], k=self.vector_db_config.similarity_top_k,
                                 embed_model=self.embed_model, splitter=text_splitter,
                                 verbose=self.vector_db_config.verbose)
            doc_results = []
            for question, answer in zip(sample.questions, sample.answers):
                retrieved_documents = self._retrieve(query=question, vector_db=vector_db, answer=answer.answer,
                                                     document_id=sample.document_id)
                doc_results.append(retrieved_documents)

                # Evaluate single question.
                question_eval_result = self.question_evaluator.evaluate(
                    sample.document_id,
                    question,
                    answer=answer.answer,
                    retriever_result=retrieved_documents,
                    k=self.vector_db_config.similarity_top_k,
                )
                self._save_eval_result(eval_result=question_eval_result, file_path=directory.replace(".json", "_question_eval.json"))
                question_eval_results.append(question_eval_result)

            document_eval_result = self.document_evaluator.evaluate(eval_sample=sample, retrieved_paragraphs=doc_results,
                                             k=self.vector_db_config.similarity_top_k)
            document_eval_results.append(document_eval_result)
            # self.save(eval_result=result, file_path=directory) # todo: uncomment this line to save results
        document_averages = self._compute_average_recall(document_eval_results)
        question_averages = self._compute_average_recall(question_eval_results)
        self._save_average_recall(filename=directory.replace(".json", "_average_scores.json"), doc_result=document_averages)
        self._save_average_recall(filename=directory.replace(".json", "_question_average_scores.json"), doc_result=question_averages)
        return document_eval_results

    def _retrieve(self, query: str, vector_db: VectorDB, answer: str, document_id: str) -> RetrieverResult:
        retriever_result = vector_db.retrieve(query=query)
        retrieved_paragraphs = RetrieverResult(
            document_id=document_id,
            question=query,
            answer=answer,
            paragraphs=[result.text for result in retriever_result],
            scores=[result.score for result in retriever_result],
        )
        return retrieved_paragraphs

    def _construct_dir(self, text_splitter) -> str:
        return self.document_evaluator.evaluator_config.output_dir + f'/{current_datetime()}/chunk_size_{text_splitter.chunk_size}_{text_splitter.__class__.__name__}_{self.data_handler.__class__.__name__}.json'

    def _save_eval_result(self, file_path: str, eval_result: EvalResult):
        """ Is used both for DocumentEvalResult and QuestionEvalResult. """
        create_dir_if_not_exists(file_path)
        # Read existing results
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []
        # Append new result
        if isinstance(eval_result, DocumentEvalResult):
            saved_doc = eval_result.eval_sample.document
            eval_result.eval_sample.document = "check ID for document"
        results.append(eval_result.model_dump())
        # Write back to the same JSON file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        eval_result.eval_sample.document = saved_doc

    def _save_all_configs(self, filename: str, splitter_config: SplitterConfig):
        configs_as_dicts = [splitter_config.model_dump()] + [config.model_dump() for config in self.configs]
        create_dir_if_not_exists(filename)
        with open(filename, 'w') as f:
            json.dump(configs_as_dicts, f, indent=4)

    def _save_average_recall(self, filename: str, doc_result: AverageDocResult):
        create_dir_if_not_exists(filename)
        with open(filename, 'w') as f:
            json.dump(doc_result.model_dump(), f, indent=4)

    def _compute_average_recall(self, eval_results: List[EvalResult]) -> AverageDocResult:
        """Is used both for DocumentEvalResult and QuestionEvalResult."""
        list_of_recalls = [eval_result.recall_at_k for eval_result in eval_results]
        result = AverageDocResult(
            average_recall_at_k=mean_of_lists(list_of_recalls),
        )
        return result


def get_splitter_configs(chunk_sizes: List[int]) -> List[SplitterConfig]:
    for chunk_size in chunk_sizes:
        yield TokenSplitterConfig(chunk_size=chunk_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the service.')
    parser.add_argument('--eval_limit', type=int, default=None, help='Limit for # data points')
    parser.add_argument('--data_handler_name', type=str, default="nq", help='Data handler to use')
    parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[64, 128, 256, 512, 1024], help='List of chunk sizes to use')

    args = parser.parse_args()
    splitter_configs = get_splitter_configs(chunk_sizes=args.chunk_sizes)
    document_evaluator_config = DocumentEvaluatorConfig(eval_limit=args.eval_limit)
    question_evaluator_config = QuestionEvaluatorConfig(eval_limit=args.eval_limit)
    data_handler_config = DataHandlerConfigFactory.create(args.data_handler_name)
    embed_model_config = EmbedModelConfig()
    vector_db_config = VectorDBConfig()
    service = Service(embed_model_config=embed_model_config,
                      splitter_configs=splitter_configs,
                      document_evaluator_config=document_evaluator_config,
                      question_evaluator_config=question_evaluator_config,
                      data_handler_config=data_handler_config,
                      vector_db_config=vector_db_config,
                      )
    service.run()
