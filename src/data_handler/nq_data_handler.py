from typing import List, Tuple, Set

import lxml.html
from tqdm import tqdm

from src.data_handler.data_handler import DataHandler
from src.dto.dto import EvalSample, Answer


class NQDataHandler(DataHandler):
    dataset_name: str = "google-research-datasets/natural_questions"

    def __init__(self, minimum_context_characters: int):
        self.minimum_context_characters = minimum_context_characters

    def _extract_documents(self, dataset, document_id: int, seen_documents: Set[str], limit: int, pbar: tqdm) -> List[
        EvalSample]:
        unique_doc_ids = set()
        samples = []
        counter = len(seen_documents)
        for dataset_sample in dataset:
            pbar.update(1)
            document = dataset_sample["document"]
            doc_id = document["url"]  # Use URL as a unique document ID

            if doc_id in unique_doc_ids:
                continue
            unique_doc_ids.add(doc_id)

            doc_text = lxml.html.fromstring(document["html"]).text_content()
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
    data = data_handler.load_data(limit=None)
