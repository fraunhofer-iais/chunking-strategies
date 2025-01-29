from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

from src.data_handler.data_handler import DataHandler
from src.dto.dto import EvalSample, Answer


class SquadDataHandler(DataHandler):
    dataset_name: str = "lhoestq/squad"

    def __init__(self, minimum_context_characters: int):
        self.minimum_context_characters = minimum_context_characters

    def load_data(self, limit: int) -> List[EvalSample]:
        ds = load_dataset(self.dataset_name)
        result = []
        for dataset in ds.values():
            current_dataset = dataset.to_dict()
            current_relevant_data = self._extract_relevant_data_from_dict(dataset_dict=current_dataset)
            result.extend(current_relevant_data)
        return result

    def _extract_relevant_data_from_dict(self, dataset_dict: Dict) -> List[EvalSample]:
        seen_docs = []
        samples = []
        document_id = 1
        for idx, document in enumerate(tqdm(dataset_dict["context"])):
            if len(document) < self.minimum_context_characters: continue
            if document not in seen_docs:
                seen_docs.append(document)
                sample = EvalSample(
                    document_id=str(document_id),
                    document=document,
                    questions=[dataset_dict["question"][idx]],
                    answers=[self.get_answer(answer=dataset_dict["answers"][idx])],
                )
                document_id += 1
                samples.append(sample)
            else:
                sample.questions.append(dataset_dict["question"][idx])
                sample.answers.append(self.get_answer(answer=dataset_dict["answers"][idx]))
        return samples

    def get_answer(self, answer: dict) -> Answer:
        return Answer(
            answer=answer["text"][0],
            start=answer["answer_start"][0],
            end=answer["answer_start"][0] + len(answer["text"][0])
        )


if __name__ == '__main__':
    data_handler = SquadDataHandler(minimum_context_characters=1250)
    data = data_handler.load_data(limit=1000)
