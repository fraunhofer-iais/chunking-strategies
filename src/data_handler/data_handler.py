from abc import abstractmethod, ABC
from typing import List

from src.dto.dto import EvalSample


class DataHandler(ABC):

    dataset_name = None

    @abstractmethod
    def load_data(self) -> List[EvalSample]: ...

