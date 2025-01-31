from src.config.config import NarrativeQADataHandlerConfig, SquadDataHandlerConfig
from src.data_handler.data_handler import DataHandler
from src.data_handler.narrative_qa_data_handler import NarrativeQADataHandler
from src.data_handler.nq_data_handler import NQDataHandler
from src.data_handler.sqad_data_handler import SquadDataHandler


class DataHandlerFactory:

    @staticmethod
    def create(data_handler_config: NarrativeQADataHandlerConfig | SquadDataHandlerConfig | NQDataHandlerConfig) \
            -> DataHandler:
        if isinstance(data_handler_config, NarrativeQADataHandlerConfig):
            return NarrativeQADataHandler()
        elif isinstance(data_handler_config, SquadDataHandlerConfig):
            return SquadDataHandler(minimum_context_characters=data_handler_config.minimum_context_characters)
        elif isinstance(data_handler_config, NQDataHandlerConfig):
            return NQDataHandler()
        else:
            raise ValueError("Invalid data handler config.")
