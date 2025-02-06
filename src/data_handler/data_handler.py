from abc import abstractmethod, ABC
from typing import List, Optional

from src.dto.dto import EvalSample


class DataHandler(ABC):

    @abstractmethod
    def load_data(self, limit: int) -> List[EvalSample]:
        """
        Load dataset and return a list of EvalSample objects.
        :param limit: If this is given, only the first `limit` samples are loaded.
        """
        ...
