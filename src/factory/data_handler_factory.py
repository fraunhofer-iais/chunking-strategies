from src.config.config import (NarrativeQADataHandlerConfig,
                               StitchedSquadDataHandlerConfig, NQDataHandlerConfig, \
                               StitchedTechQADataHandlerConfig, StitchedNewsQADataHandlerConfig,
                               HybridDataHandlerConfig, DataHandlerConfig)
from src.data_handler.data_handler import DataHandler
from src.data_handler.hybrid_data_handler import HybridDataHandler
from src.data_handler.narrative_qa_data_handler import NarrativeQADataHandler
from src.data_handler.nq_data_handler import NQDataHandler
from src.data_handler.stitched_newsqa_data_handler import StitchedNewsQADataHandler
from src.data_handler.stitched_squad_data_handler import StitchedSquadDataHandler
from src.data_handler.stitched_tech_qa_data_handler import StitchedTechQADataHandler


class DataHandlerFactory:

    @staticmethod
    def create(data_handler_config: DataHandlerConfig) \
            -> DataHandler:
        if isinstance(data_handler_config, NarrativeQADataHandlerConfig):
            return NarrativeQADataHandler()
        elif isinstance(data_handler_config, StitchedSquadDataHandlerConfig):
            return StitchedSquadDataHandler(minimum_context_characters=data_handler_config.minimum_context_characters)
        elif isinstance(data_handler_config, NQDataHandlerConfig):
            return NQDataHandler(minimum_context_characters=data_handler_config.minimum_context_characters)
        elif isinstance(data_handler_config, StitchedTechQADataHandlerConfig):
            return StitchedTechQADataHandler(minimum_context_characters=
                                                   data_handler_config.minimum_context_characters)
        elif isinstance(data_handler_config, StitchedNewsQADataHandlerConfig):
            return StitchedNewsQADataHandler(minimum_context_characters=data_handler_config.minimum_context_characters)
        elif isinstance(data_handler_config, HybridDataHandlerConfig):
            return HybridDataHandler(data_handler_config)
        else:
            raise ValueError("Invalid data handler config.")
