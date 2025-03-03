import os
import pickle
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from typing import List, Set

from datasets import load_dataset
from tqdm import tqdm

from src.dto.dto import EvalSample


class DataHandler(ABC):
    dataset_name: str = None

    def load_data(self, limit: int) -> List[EvalSample]:
        """
        Load dataset and return a list of EvalSample objects.
        :param limit: If this is given, only the first `limit` samples are loaded.
        """
        cached_data = self._get_from_cache()
        if cached_data:
            if limit and len(cached_data) >= limit:
                print(f"Cached dataset with len {len(cached_data)} is used. This is above the limit of {limit}, so we cut it off.")
                return cached_data[:limit]
            else:
                print(f"Cached dataset with len {len(cached_data)} is used.")
                return cached_data

        ds = load_dataset(self.dataset_name, streaming=True)
        result = []
        document_id = 1  # Unique document ID counter
        seen_documents: Set[str] = set()  # Store seen documents to avoid duplicates
        counter = 0

        total_length = ds["train"].info.splits.total_num_examples

        with ThreadPoolExecutor() as executor, tqdm(total=total_length, desc="Processing dataset") as pbar:
            futures = [
                executor.submit(
                    self._extract_documents,
                    dataset=dataset,
                    document_id=document_id,
                    seen_documents=seen_documents,
                    limit=limit - counter if limit is not None else None,
                    pbar=pbar
                )
                for dataset in ds.values()
            ]

            for future in futures:
                samples = future.result()
                result.extend(samples)
                document_id += len(samples)
                counter += len(samples)
                if limit is not None and counter >= limit:
                    break

        self._save_to_cache(data=result)
        return result

    @abstractmethod
    def _extract_documents(self, dataset, document_id: int, seen_documents: Set[str], limit: int, pbar: tqdm) \
            -> List[EvalSample]:
        ...

    def _get_from_cache(self) -> List[EvalSample]:
        cache_file = self._cache_dir()
        if not os.path.exists(cache_file):
            return []
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def _save_to_cache(self, data: List[EvalSample]):
        cache_file = self._cache_dir()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    def _cache_dir(self) -> str:
        directory = os.getenv("HF_HOME")
        if directory is None:
            raise ValueError("HF_HOME environment variable is not set.")
        class_name = self.__class__.__name__
        processed_directory = directory + "/processed_chunking_datasets/"
        os.makedirs(processed_directory, exist_ok=True)
        return processed_directory + class_name + ".pkl"
