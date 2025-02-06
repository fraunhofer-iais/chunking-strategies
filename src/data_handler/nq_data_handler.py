from typing import List, Tuple, Optional

from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm

from src.data_handler.data_handler import DataHandler
from src.dto.dto import EvalSample, Answer


class NQDataHandler(DataHandler):
    dataset_name: str = "google-research-datasets/natural_questions"

    def __init__(self, minimum_context_characters: int):
        self.minimum_context_characters = minimum_context_characters

    def load_data(self, limit: Optional[int] = None) -> List[EvalSample]:
        ds = load_dataset(self.dataset_name, streaming=True)
        result = []
        counter = 0

        for dataset in ds.values():
            current_relevant_data = self._extract_relevant_data_from_dict(dataset, counter, limit)
            result.extend(current_relevant_data)
            counter += len(current_relevant_data)
            if limit and counter >= limit:
                break
        return result

    def _extract_relevant_data_from_dict(self, dataset, counter: int, limit: int) -> List[EvalSample]:
        unique_doc_ids = set()
        samples = []

        for dataset_sample in tqdm(dataset):
            document = dataset_sample["document"]
            doc_id = document["url"]  # Use URL as a unique document ID

            if doc_id in unique_doc_ids:
                continue
            unique_doc_ids.add(doc_id)

            doc_text = BeautifulSoup(document["html"], "html.parser").get_text()
            if len(doc_text) <= self.minimum_context_characters:
                continue

            short_answers = dataset_sample["annotations"]["short_answers"]  # Ensure annotations exist
            if not short_answers or "text" not in short_answers[0] or not short_answers[0]["text"]:
                continue  # Skip samples with no valid short answer

            answer_text = short_answers[0]["text"][0]

            span = self._get_span(doc=doc_text, answer=answer_text)
            if not span:
                continue

            sample = EvalSample(document_id=doc_id, document=doc_text)
            sample.questions = [dataset_sample["question"]["text"]]
            sample.answers = [Answer(answer=answer_text, start=span[0], end=span[1])]

            samples.append(sample)
            counter += 1

            if limit and counter >= limit:
                break

        return samples

    def _get_span(self, doc: str, answer: str) -> Tuple[int, int] | None:
        """ Ensures only one extractive span per document """
        spans = []
        start = 0
        while start < len(doc):
            start_index = doc.lower().find(answer.lower(), start)
            if start_index == -1:
                break
            end_index = start_index + len(answer)
            spans.append((start_index, end_index))
            start = end_index

        if len(spans) != 1:
            return None

        return spans[0]


if __name__ == "__main__":
    data_handler = NQDataHandler(minimum_context_characters=50000)
    data = data_handler.load_data(limit=5)
    print(data)
