import json
import os
from datetime import datetime
from typing import List, Any

from llama_index.core import Document
from tqdm import tqdm

from src.config.config import ServiceConfig, TokenSplitterConfig, SemanticSplitterConfig, SentenceSplitterConfig, \
    EvaluatorConfig, EmbedModelConfig
from src.dto.dto import RetrievedParagraphs, EvalSample, RetrieverResults
from src.factory.embed_model_factory import EmbedModelFactory
from src.factory.evaluator_factory import EvaluatorFactory
from src.factory.splitter_factory import SplitterFactory
from src.vector_db.vector_db import VectorDB


def current_datetime(fmt: str = "%m%d%Y_%H%M%S") -> str:
    now = datetime.now()
    date_time = now.strftime(fmt)
    return date_time


class Service:
    def __init__(self,
                 service_config: ServiceConfig,
                 splitter_config: TokenSplitterConfig | SentenceSplitterConfig | SemanticSplitterConfig,
                 evaluator_config: EvaluatorConfig,
                 embed_model_config: EmbedModelConfig,
                 ):
        self.service_config = service_config
        self.evaluator = EvaluatorFactory.create(evaluator_config=evaluator_config)
        self.text_splitter = SplitterFactory.create(splitter_config=splitter_config)
        self.embed_model = EmbedModelFactory.create(embed_model_config=embed_model_config)

    def run(self) -> dict[str, list[Any]]:
        data: List[EvalSample] = self.evaluator.evaluator_config.data_handler.load_data(
            limit=self.evaluator.evaluator_config.eval_limit)
        results = {
            "samples": [],
            "retrieved_paragraphs": []
        }
        for sample in tqdm(data[self.evaluator.evaluator_config.eval_start:self.evaluator.evaluator_config.eval_limit]):
            doc = Document(text=sample.document, doc_id=sample.document_id)
            vector_db = VectorDB(documents=[doc], k=self.service_config.similarity_top_k,
                                 embed_model=self.embed_model, splitter=self.text_splitter,
                                 verbose=self.service_config.vector_db_verbose)
            doc_results = []
            for question, answer in zip(sample.questions, sample.answers):
                retrieved_documents = self._retrieve(query=question, vector_db=vector_db, answer=answer.answer,
                                                     document_id=sample.document_id)
                doc_results.append(retrieved_documents)
            results["samples"].append(sample)
            results["retrieved_paragraphs"].append(doc_results)
            self.evaluator.evaluate(eval_sample=sample, retrieved_paragraphs=doc_results)
            ...
        return results

    def _retrieve(self, query: str, vector_db: VectorDB, answer: str, document_id: str) -> RetrievedParagraphs:
        retriever_result = vector_db.retrieve(query=query)
        retrieved_paragraphs = RetrievedParagraphs(
            document_id=document_id,
            question=query,
            answer=answer,
            paragraphs=[result.text for result in retriever_result],
            scores=[result.score for result in retriever_result],
        )
        return retrieved_paragraphs

    def evaluate(self, evaluation_data: dict[str, List[Any]]) -> RetrieverResults:
        samples = evaluation_data["samples"]
        retrieved_paragraphs = evaluation_data["retrieved_paragraphs"]
        results = self.evaluator.evaluate_multiple_documents(eval_samples=samples, predictions=retrieved_paragraphs)
        return results

    def save(self, results: RetrieverResults):
        results_dict = results.model_dump()

        directory = self.evaluator.evaluator_config.output_dir + f'/{current_datetime("%m%d%Y")}/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        os.makedirs(self.evaluator.evaluator_config.output_dir, exist_ok=True)
        data_handler_name = self.evaluator.evaluator_config.data_handler.__class__.__name__

        file_path = os.path.join(directory,
                                 f'chunk_size_{self.text_splitter.chunk_size}_splitter_{self.text_splitter.__class__.__name__}'
                                 f'_data_{data_handler_name}_{current_datetime("%H%M%S")}.json')

        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {file_path}")


if __name__ == '__main__':
    config = ServiceConfig()
    splitter_config = TokenSplitterConfig()
    evaluator_config = EvaluatorConfig()
    embed_model_config = EmbedModelConfig()
    service = Service(embed_model_config=embed_model_config, service_config=config, splitter_config=splitter_config,
                      evaluator_config=evaluator_config)
    responses = service.run()
    evaluation_responses = service.evaluate(responses)
    service.save(evaluation_responses)
