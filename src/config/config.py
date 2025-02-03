from typing import List, Union, Optional

from pydantic import BaseModel

from src.constants.constants import STELLA_EN_1_5B_V5


class VectorDBConfig(BaseModel):
    similarity_top_k: int = 5  # how many chunks should we retrieve?
    verbose: bool = False


class EmbedModelConfig(BaseModel):
    embed_model_name: str = STELLA_EN_1_5B_V5
    embed_model_device: str = "cuda"


class SemanticSplitterConfig(BaseModel):
    buffer_size: int = 5  # number of sentences to group together when evaluating semantic similarity
    breakpoint_percentile_threshold: int = 95  # The percentile of cosine dissimilarity that must be exceeded between
    # a group of sentences and the next to form a node.  The smaller this number is, the more nodes will be generated.


class TokenSplitterConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int = None
    separator: str = ' '


class SentenceSplitterConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int = None
    include_metadata: bool = True
    include_prev_next_rel: bool = True
    separator: str = ' '
    paragraph_separator: str = '\n\n\n'
    secondary_chunking_regex: str = '[^,.;。？！]+[,.;。？！]?'


class EvaluatorConfig(BaseModel):
    eval_start: int = None
    eval_limit: int = None
    output_dir: str = "output"


class NarrativeQADataHandlerConfig(BaseModel):
    ...


class StitchedSquadDataHandlerConfig(BaseModel):
    minimum_context_characters: int = 50000 # minimum number of characters in a context to be considered for evaluation


class StitchedTechQADataHandlerConfig(BaseModel):
    minimum_context_characters: int = 50000


class NQDataHandlerConfig(BaseModel):
    minimum_context_characters: int = 50000


class StitchedNewsQADataHandlerConfig(BaseModel):
    minimum_context_characters: int = 50000


class HybridDataHandlerConfig(BaseModel):
    handler_configs: List[Union[
    StitchedSquadDataHandlerConfig,
    StitchedTechQADataHandlerConfig,
    StitchedNewsQADataHandlerConfig,
    NQDataHandlerConfig,
    NarrativeQADataHandlerConfig]] = \
        [StitchedSquadDataHandlerConfig, StitchedTechQADataHandlerConfig] # List of individual handler configurations
    limit_samples_per_dataset: int = 2 # maximum number of samples to be processed from each individual dataset

