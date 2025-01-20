from typing import Iterator
from urllib.request import DataHandler
from datasets import load_dataset, Dataset
from tqdm import tqdm

from src.dto.dto import EvalSample, Answer, Span


class HotpotQADataHandler(DataHandler):
    def __init__(self, remove_incomplete_samples: bool = True):
        """
        :param remove_incomplete_samples: If True, samples with just 1 span information are removed.

        About 2000 out of 8000 samples are removed when setting remove_incomplete_samples to True.
        """
        super().__init__()
        self.remove_incomplete_samples = remove_incomplete_samples

    dataset_name : str = "TIGER-Lab/LongRAG"
    subset: str = "hotpot_qa"
    split: str = "full"  # "full" | "subset_1000"

    def load_data(self) -> list[EvalSample]:
        dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split, streaming=True)
        return self._extract_relevant_data_from_dict(dataset)

    def _extract_relevant_data_from_dict(self, dataset: Dataset) -> Iterator[EvalSample]:
        for sample in dataset:
            spans = self._get_answer_spans(document=sample["context"], span_names=sample["sp"])

            if spans is not None:
                eval_sample = EvalSample(document_id=str(sample["query_id"]), document=sample["context"])
                eval_sample.questions = [sample["query"]]

                answers = [Answer(answer=sample["answer"][0], spans=spans)]
                eval_sample.answers = answers
                yield eval_sample

    def _get_answer_spans(self, document: dict, span_names: list[str]) -> list[Span] | None:
        spans = []

        title_str = "Title: "
        title_str_len = len(title_str)

        for span_name in span_names:
            title_plus_span_name = title_str + span_name + "\n"

            # Check that the span identifier has a unique occurrence in the document.
            if document.count(title_plus_span_name) < 1:
                if self.remove_incomplete_samples:
                    return None
                else:
                    continue
            elif document.count(title_plus_span_name) > 1:
                raise ValueError("The span name is found multiple times in the document.")

            title_plus_span_name_start = document.index(title_plus_span_name)
            title_plus_span_name_end = title_plus_span_name_start + len(title_plus_span_name)

            start = title_plus_span_name_end
            end = start + 1
            for _char in document[start:]:
                if document[end:end + title_str_len] == title_str:
                    break
                end += 1

            spans.append(Span(start=start, end=end))
        return spans


if __name__ == "__main__":

    data_handler = HotpotQADataHandler()
    data = data_handler.load_data()

    for sample in tqdm(data):
        print(sample)
