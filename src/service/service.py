import argparse
import json
from typing import List

from llama_index.core import Document
from tqdm import tqdm

from src.config.config import TokenSplitterConfig, SemanticSplitterConfig, SentenceSplitterConfig, \
    EvaluatorConfig, EmbedModelConfig, NarrativeQADataHandlerConfig, StitchedSquadDataHandlerConfig, VectorDBConfig, \
    NQDataHandlerConfig, StitchedNewsQADataHandlerConfig, StitchedTechQADataHandlerConfig, HybridDataHandlerConfig, \
    DataHandlerConfig
from src.data_handler.stitched_squad_data_handler import StitchedSquadDataHandler
from src.dto.dto import RetrieverResult, EvalSample, EvalResult, AverageDocResult
from src.factory.data_handler_factory import DataHandlerFactory
from src.factory.embed_model_factory import EmbedModelFactory
from src.factory.evaluator_factory import EvaluatorFactory
from src.factory.splitter_factory import SplitterFactory
from src.utils import mean_of_lists, create_dir_if_not_exists
from src.vector_db.vector_db import VectorDB


class Service:
    def __init__(self,
                 splitter_config: TokenSplitterConfig | SentenceSplitterConfig | SemanticSplitterConfig,
                 evaluator_config: EvaluatorConfig,
                 embed_model_config: EmbedModelConfig,
                 data_handler_config: DataHandlerConfig,
                 vector_db_config: VectorDBConfig,
                 ):
        self.configs = [
            splitter_config,
            evaluator_config,
            embed_model_config,
            data_handler_config,
            vector_db_config,
        ]
        self.vector_db_config = vector_db_config
        self.evaluator = EvaluatorFactory.create(evaluator_config=evaluator_config)
        self.text_splitter = SplitterFactory.create(splitter_config=splitter_config)
        self.embed_model = EmbedModelFactory.create(embed_model_config=embed_model_config)
        self.data_handler = DataHandlerFactory.create(data_handler_config=data_handler_config)

    def run(self) -> List[EvalResult]:
        data: List[EvalSample] = self.data_handler.load_data(limit=self.evaluator.evaluator_config.eval_limit)
        eval_results = []
        directory = self._construct_dir()
        self._save_all_configs(filename=directory.replace(".json", "_configs.json"))
        for sample in tqdm(data[self.evaluator.evaluator_config.eval_start:self.evaluator.evaluator_config.eval_limit]):
            doc = Document(text=sample.document, doc_id=sample.document_id)
            vector_db = VectorDB(documents=[doc], k=self.vector_db_config.similarity_top_k,
                                 embed_model=self.embed_model, splitter=self.text_splitter,
                                 verbose=self.vector_db_config.verbose)
            doc_results = []
            for question, answer in zip(sample.questions, sample.answers):
                retrieved_documents = self._retrieve(query=question, vector_db=vector_db, answer=answer.answer,
                                                     document_id=sample.document_id)
                doc_results.append(retrieved_documents)
            result = self.evaluator.evaluate(eval_sample=sample, retrieved_paragraphs=doc_results)
            eval_results.append(result)
            self.save(eval_result=result, file_path=directory)
        averages = self._compute_average_recall(eval_results)
        self._save_average_recall(filename=directory.replace(".json", "_average_scores.json"), doc_result=averages)
        return eval_results

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

    def _construct_dir(self):
        return self.evaluator.evaluator_config.output_dir + f'/chunk_size_{self.text_splitter.chunk_size}_{self.text_splitter.__class__.__name__}_{self.data_handler.__class__.__name__}.json'

    def save(self, file_path: str, eval_result: EvalResult):
        create_dir_if_not_exists(file_path)
        # Read existing results
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []
        # Append new result
        eval_result.eval_sample.document = "check ID for document"
        results.append(eval_result.model_dump())
        # Write back to the same JSON file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _save_all_configs(self, filename: str):
        configs_as_dicts = [config.model_dump() for config in self.configs]
        create_dir_if_not_exists(filename)
        with open(filename, 'w') as f:
            json.dump(configs_as_dicts, f, indent=4)

    def _save_average_recall(self, filename: str, doc_result: AverageDocResult):
        create_dir_if_not_exists(filename)
        with open(filename, 'w') as f:
            json.dump(doc_result.model_dump(), f, indent=4)

    def _compute_average_recall(self, eval_results: List[EvalResult]) -> AverageDocResult:
        list_of_recalls = [eval_result.recall_at_k for eval_result in eval_results]
        result = AverageDocResult(
            average_recall_at_k=mean_of_lists(list_of_recalls),
        )
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the service with configurable chunk size.')
    parser.add_argument('--chunk_size', type=int, default=64, help='Chunk size for the splitter')
    args = parser.parse_args()

    splitter_config = TokenSplitterConfig(chunk_size=args.chunk_size)
    evaluator_config = EvaluatorConfig()
    embed_model_config = EmbedModelConfig()
    #data_handler_config = NarrativeQADataHandlerConfig()

    data_handler_config = HybridDataHandlerConfig()
    vector_db_config = VectorDBConfig()
    service = Service(embed_model_config=embed_model_config,
                      splitter_config=splitter_config,
                      evaluator_config=evaluator_config,
                      data_handler_config=data_handler_config,
                      vector_db_config=vector_db_config,
                      )
    responses = service.run()
