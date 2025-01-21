import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

from src.config.config import ServiceConfig, TokenSplitterConfig, SemanticSplitterConfig, SentenceSplitterConfig, \
    LoggingConfig, EvaluatorConfig
from src.dto.dto import RetrievedParagraphs
from src.factory.splitter_factory import SplitterFactory
from src.vector_db.vector_db import VectorDB


def read_from_pdf(data_dir: str) -> List[Document]:
    documents = SimpleDirectoryReader(data_dir).load_data()
    return documents


def read_from_xlsx(data_dir: str) -> Dict[str, Any]:
    result = pd.read_excel(data_dir)
    return result.to_dict()


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

    def run(self) -> List[List[RetrievedParagraphs]]:
        data = evaluator_config.data_handler.load_data()
        results = []
        for sample in tqdm(data[:evaluator_config.eval_limit]):
            doc = Document(text=sample.document, doc_id=sample.document_id)
            vector_db = VectorDB(documents=[doc], k=self.config.similarity_top_k,
                                 embed_model=self.embed_model, splitter=self.text_splitter)
            doc_results = []
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
            results.append(doc_results)
        # todo save result
        # result_dict = responses.model_dump()
        # self._save_result(result_dict, chunk_size=self.config.chunk_size)
        return results

    def _save_result(self, result_dict: dict, chunk_size: int):
        directory = self.config.out_dir + f'{current_datetime("%m%d%Y")}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # fixme
        filename = f'{chunk_size}_{current_datetime("%H%M%S")}.json'
        with open(directory + filename, 'w') as ff:
            json.dump(result_dict, ff)


if __name__ == '__main__':
    config = ServiceConfig()
    splitter_config = TokenSplitterConfig()
    splitter_factory = SplitterFactory()
    logging_config = LoggingConfig()
    evaluator_config = EvaluatorConfig()
    chunker = Service(config=config, splitter_factory=splitter_factory, splitter_config=splitter_config,
                      logging_config=logging_config, evaluator_config=evaluator_config)
    responses = chunker.run()
    ...
