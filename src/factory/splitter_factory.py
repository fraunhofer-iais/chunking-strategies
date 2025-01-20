from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, NodeParser, TokenTextSplitter

from src.config.config import TokenSplitterConfig, SentenceSplitterConfig, SemanticSplitterConfig


class SplitterFactory:

    @staticmethod
    def create(splitter_config: SemanticSplitterConfig | TokenSplitterConfig | SentenceSplitterConfig) -> NodeParser:
        if hasattr(splitter_config, "chunk_overlap") and splitter_config.chunk_overlap is None:
            splitter_config.chunk_overlap = splitter_config.chunk_size // 5
        if isinstance(splitter_config, TokenSplitterConfig):
            return TokenTextSplitter(
                chunk_size=splitter_config.chunk_size,
                chunk_overlap=splitter_config.chunk_overlap,
                separator=splitter_config.separator,
            )
        elif isinstance(splitter_config, SentenceSplitterConfig):
            return SentenceSplitter(include_metadata=splitter_config.include_metadata,
                                    include_prev_next_rel=splitter_config.include_prev_next_rel,
                                    chunk_size=splitter_config.chunk_size,
                                    chunk_overlap=splitter_config.chunk_overlap,
                                    separator=splitter_config.separator,
                                    paragraph_separator=splitter_config.paragraph_separator,
                                    secondary_chunking_regex=splitter_config.secondary_chunking_regex,
                                    )
        elif isinstance(splitter_config, SemanticSplitterConfig):
            return SemanticSplitterNodeParser(
                include_metadata=splitter_config.include_metadat,
                buffer_size=splitter_config.semantic_chunking_config.buffer_size,
                breakpoint_percentile_threshold=splitter_config.semantic_chunking_config.breakpoint_percentile_threshold,
                embed_model=Settings.embed_model,
            )
        else:
            raise ValueError(f"{splitter_config} is not supported.")