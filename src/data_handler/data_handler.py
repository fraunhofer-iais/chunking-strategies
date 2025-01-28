from abc import abstractmethod, ABC
from typing import List

from src.dto.dto import EvalSample


class DataHandler(ABC):

    @abstractmethod
    def load_data(self, limit: int) -> List[EvalSample]: ...
