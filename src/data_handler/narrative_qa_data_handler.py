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
            if doc_id not in unique_doc_ids:
                sample = EvalSample(document_id=doc_id, document=document["text"])
                sample.questions = [dataset_dict["question"][idx]["text"]]
                sample.answers = [Answer(answer=dataset_dict["answers"][idx][0]["text"])]
                # todo find text span (not clearly defined here)
                ...
                unique_doc_ids.append(doc_id)
                samples.append(sample)
            else:
                sample.questions.append(dataset_dict["question"][idx]["text"])
                sample.answers.append(Answer(answer=dataset_dict["answers"][idx][0]["text"]))
        return samples


if __name__ == "__main__":
    data_handler = NarrativeQADataHandler()
    data = data_handler.load_data()
    ...
