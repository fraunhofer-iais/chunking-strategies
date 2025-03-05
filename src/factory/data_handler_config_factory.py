from typing import Literal

from src.config.config import DataHandlerConfig, HybridDataHandlerConfig, NarrativeQADataHandlerConfig, \
    NQDataHandlerConfig, StitchedCovidQADataHandlerConfig, StitchedNewsQADataHandlerConfig, \
    StitchedSquadDataHandlerConfig, StitchedTechQADataHandlerConfig


class DataHandlerConfigFactory:

    @staticmethod
    def create(data_handler_name: Literal[
        "hybrid", "narrative_qa", "nq", "stitched_covid_qa", "stitched_newsqa", "stitched_squad", "stitched_tech_qa"
    ]) -> DataHandlerConfig:
        match data_handler_name:
            case "hybrid":
                return HybridDataHandlerConfig()
            case "narrative_qa":
                return NarrativeQADataHandlerConfig()
            case "nq":
                return NQDataHandlerConfig()
            case "stitched_covid_qa":
                return StitchedCovidQADataHandlerConfig()
            case "stitched_newsqa":
                return StitchedNewsQADataHandlerConfig()
            case "stitched_squad":
                return StitchedSquadDataHandlerConfig()
            case "stitched_tech_qa":
                return StitchedTechQADataHandlerConfig()
            case _:
                raise ValueError(f"Data handler name {data_handler_name} is not supported.")
