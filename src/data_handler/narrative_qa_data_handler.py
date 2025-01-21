from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

from src.data_handler.data_handler import DataHandler
from src.dto.dto import EvalSample, Answer


class NarrativeQADataHandler(DataHandler):
    dataset_name: str = "deepmind/narrativeqa"

    def load_data(self) -> List[EvalSample]:
        ds = load_dataset(self.dataset_name)
        result = []
        for dataset in ds.values():
            current_dataset = dataset.to_dict()
            current_relevant_data = self._extract_relevant_data_from_dict(dataset_dict=current_dataset)
            result.extend(current_relevant_data)
        return result

    def _extract_relevant_data_from_dict(self, dataset_dict: Dict) -> List[EvalSample]:
        unique_doc_ids = []
        samples = []
        for idx, document in enumerate(tqdm(dataset_dict["document"])):
            doc_id = document["id"]
            doc = document["text"]
            answer = dataset_dict["answers"][idx][0]["text"]
            span = self.get_span(doc=doc, answer=answer)
            if not span:
                continue
            if doc_id not in unique_doc_ids:
                sample = EvalSample(document_id=doc_id, document=doc)
                sample.questions = [dataset_dict["question"][idx]["text"]]
                sample.answers = [Answer(answer=answer, start=span[0], end=span[1])]
                unique_doc_ids.append(doc_id)
                samples.append(sample)
            else:
                sample.questions.append(dataset_dict["question"][idx]["text"])
                sample.answers.append(Answer(answer=answer, start=span[0], end=span[1]))
        return samples

    def get_answer(self, answer: str, context: str) -> Answer:
        start = context.find(answer)
        end = start + len(answer)
        return Answer(answer=answer, start=start, end=end)

    def answer_in_context(self, answer: str, context: str) -> bool:
        return answer.lower() in context.lower()

    def get_all_spans(self, text: str, substring: str) -> List[tuple]:
        spans = []
        start = 0
        while start < len(text):
            start_index = text.find(substring, start)
            if start_index == -1:
                break
            end_index = start_index + len(substring)
            spans.append((start_index, end_index))
            start = end_index  # Continue searching after this occurrence
        return spans

    def get_span(self, doc: str, answer: str) -> tuple | None:
        if not self.answer_in_context(answer=answer, context=doc):
            return None
        spans = self.get_all_spans(text=doc.lower(), substring=answer.lower())
        if len(spans) > 1:
            return None
        return spans[0]


if __name__ == "__main__":
    data_handler = NarrativeQADataHandler()
    data = data_handler.load_data()
    ...
