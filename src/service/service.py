import json
import logging
import os
from datetime import datetime
from typing import List, Any

from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

from src.config.config import ServiceConfig, TokenSplitterConfig, SemanticSplitterConfig, SentenceSplitterConfig, \
    LoggingConfig, EvaluatorConfig
from src.dto.dto import RetrievedParagraphs, EvalSample, RetrieverResults
from src.evaluator.evaluate import Evaluator
from src.factory.splitter_factory import SplitterFactory
from src.vector_db.vector_db import VectorDB


def current_datetime(fmt: str = "%m%d%Y_%H%M%S") -> str:
    now = datetime.now()
    date_time = now.strftime(fmt)
    return date_time


class Service:
    def __init__(self,
                 config: ServiceConfig,
                 splitter_factory: SplitterFactory,
                 splitter_config: TokenSplitterConfig | SentenceSplitterConfig | SemanticSplitterConfig,
                 logging_config: LoggingConfig,
                 evaluator_config: EvaluatorConfig,
                 ):
        self.config = config
        self.logging_config = logging_config
        self.evaluator_config = evaluator_config
        self.splitter_config = splitter_config

        self.splitter_factory = splitter_factory

        self.text_splitter = self._initialize_text_splitter()
        self.embed_model = self._initialize_embed_model()

    def _setup_logging(self):
        logging.basicConfig(
            level=self.config.level,
            format=self.config.format,
        )
        return logging.getLogger(__name__)

    def _initialize_embed_model(self):
        return HuggingFaceEmbedding(
            model_name=self.config.embed_model_name,
            device=self.config.embed_model_device,
            trust_remote_code=True
        )

    def _initialize_text_splitter(self):
        return self.splitter_factory.create(splitter_config=self.splitter_config)

    def run(self) -> dict[str, list[Any]]:

        data: List[EvalSample] = evaluator_config.data_handler.load_data()
        results = {
            "samples": [],
            "retrieved_paragraphs": []
        }
        for sample in tqdm(data[:evaluator_config.eval_limit]):
            doc = Document(text=sample.document, doc_id=sample.document_id)
            vector_db = VectorDB(documents=[doc], k=self.config.similarity_top_k,
                                 embed_model=self.embed_model, splitter=self.text_splitter)
            doc_results = []
            results["samples"].append(sample)
            for question, answer in zip(sample.questions, sample.answers):
                result = vector_db.retrieve(query=question)
                retrieved_documents = RetrievedParagraphs(
                    document_id=sample.document_id,
                    question=question,
                    answer=answer.answer,
                    paragraphs=[result.text for result in result],
                    scores=[result.score for result in result],
                )
                doc_results.append(retrieved_documents)
            results["retrieved_paragraphs"].append(doc_results)
        return results

    def evaluate(self, evaluation_data: dict[str, List[Any]]) -> RetrieverResults:
        samples = evaluation_data["samples"]

        retrieved_paragraphs = evaluation_data["retrieved_paragraphs"]
        evaluator = Evaluator(self.evaluator_config)

        results = evaluator.evaluate_multiple_documents(eval_samples=samples, predictions=retrieved_paragraphs)
        return results

    def save(self, results: RetrieverResults):
        results_dict = results.model_dump()

        directory = self.evaluator_config.output_dir + f'{current_datetime("%m%d%Y")}/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        os.makedirs(self.evaluator_config.output_dir, exist_ok=True)
        data_handler_name = self.evaluator_config.data_handler.__class__.__name__
        dataset_name = data_handler_name.split('/')[-1]

        file_path = os.path.join(directory,
                                 f'chunk_size_{self.splitter_config.chunk_size}_splitter_{self.splitter_config.__class__.__name__}'
                                 f'_data_{dataset_name}_{current_datetime("%H%M%S")}.json')

        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {file_path}")


if __name__ == '__main__':
    config = ServiceConfig()
    splitter_config = TokenSplitterConfig()
    splitter_factory = SplitterFactory()
    logging_config = LoggingConfig()
    evaluator_config = EvaluatorConfig()
    chunker = Service(config=config, splitter_factory=splitter_factory, splitter_config=splitter_config,
                      logging_config=logging_config, evaluator_config=evaluator_config)
    responses = chunker.run()
    evaluation_responses = chunker.evaluate(responses)
    chunker.save(evaluation_responses)
