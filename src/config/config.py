from typing import Type, Literal

from pydantic import BaseModel, BaseConfig

from src.constants.constants import GPT4O, STELLA_EN_1_5B_V5
from src.data_handler.data_handler import DataHandler
from src.data_handler.narrative_qa_data_handler import NarrativeQADataHandler
from src.data_handler.sqad_data_handler import SquadDataHandler


class ServiceConfig(BaseModel):
    embed_model_name: str = STELLA_EN_1_5B_V5
    embed_model_device: str = "cuda"
    similarity_top_k: int = 5  # how many chunks should we retrieve?
    vector_db_verbose: bool = False


class SemanticSplitterConfig(BaseModel):
    buffer_size: int = 5  # number of sentences to group together when evaluating semantic similarity
    breakpoint_percentile_threshold: int = 95  # The percentile of cosine dissimilarity that must be exceeded between
    # a group of sentences and the next to form a node.  The smaller this number is, the more nodes will be generated.


class TokenSplitterConfig(BaseModel):
    chunk_size: int = 128
    chunk_overlap: int = None
    separator: str = ' '


class SentenceSplitterConfig(BaseModel):
    chunk_size: int = 3
    chunk_overlap: int = None
    include_metadata: bool = True
    include_prev_next_rel: bool = True
    separator: str = ' '
    paragraph_separator: str = '\n\n\n'
    secondary_chunking_regex: str = '[^,.;。？！]+[,.;。？！]?'


class EvaluatorConfig(BaseModel):
    eval_limit: int = None
    data_handler: Type[DataHandler] = NarrativeQADataHandler()
    k: int = 5
    output_dir: str = "output"


class LoggingConfig(BaseModel):  # LoggingConfig
    level: str = "INFO"  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    filename: str = None  # If specified, logs will be written to a file
