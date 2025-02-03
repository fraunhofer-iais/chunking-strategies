import random
from typing import List, Dict, Set, Tuple

from datasets import load_dataset
from tqdm import tqdm

from src.data_handler.data_handler import DataHandler
from src.dto.dto import EvalSample, Answer


class StitchedTechQADataHandler(DataHandler):
    dataset_name: str = "rojagtap/tech-qa"

    def __init__(self, minimum_context_characters: int):
        self.minimum_context_characters = minimum_context_characters

    def load_data(self, limit: int) -> List[EvalSample]:
        ds = load_dataset(self.dataset_name, streaming=True)
        result = []
        document_id = 1  # Unique document ID counter
        seen_documents: Set[str] = set()  # Store seen documents to avoid duplicates
        counter = 0

        for dataset in ds.values():
            current_dataset = dataset.to_dict()
            stitched_samples = self._extract_and_stitch_documents(
                dataset_dict=current_dataset,
                document_id=document_id,
                seen_documents=seen_documents,
                limit = limit - counter
            )
            result.extend(stitched_samples)
            document_id += len(stitched_samples)
            counter += len(stitched_samples)

            if limit and counter >= limit:
                break

        return result

    def _extract_and_stitch_documents(self, dataset_dict: Dict, document_id: int, seen_documents: Set[str], limit: int) \
            -> List[EvalSample]:
        buffer = []  # Store short documents for stitching
        samples = []

        for idx, document in enumerate(tqdm(dataset_dict["document"])):
            if limit and len(samples) >= limit:
                break
            question = dataset_dict["question"][idx]
            answer_data = dataset_dict["answer"][idx]
            answer = self.get_answer(answer_data)

            # Skip duplicate documents
            if document in seen_documents:
                continue
            seen_documents.add(document)

            # If document is long enough, use it directly
            if len(document) >= self.minimum_context_characters:
                if document.lower().count(answer.answer.lower()) == 1:
                    sample = EvalSample(
                        document_id=str(document_id),
                        document=document,
                        questions=[question],
                        answers=[answer],
                    )
                    samples.append(sample)
                    document_id += 1
            else:
                # Store short documents in buffer for stitching
                buffer.append((document, question, answer))

                # If we have enough documents in buffer, stitch them
                stitched_length = sum(len(doc[0]) for doc in buffer)
                if stitched_length >= self.minimum_context_characters:
                    stitched_doc, chosen_question, chosen_answer = self._stitch_documents(buffer)
                    buffer = []  # Reset buffer after stitching

                    sample = EvalSample(
                        document_id=f"stitched_{document_id}",
                        document=stitched_doc,
                        num_tokens=len(stitched_doc.split()),
                        num_chars=len(stitched_doc),
                        questions=[chosen_question],
                        answers=[chosen_answer],
                    )
                    samples.append(sample)
                    document_id += 1

        return samples

    def _stitch_documents(self, buffer: List[Tuple[str, str, Answer]]) -> Tuple[str, str, Answer]:
        """ Stitches multiple short documents into one and selects a random Q&A pair """
        random.shuffle(buffer)  # Shuffle to mix documents
        stitched_doc = " ".join([doc[0] for doc in buffer])  # Combine all texts
        chosen_doc = random.choice(buffer)  # Randomly select one document for Q&A
        return stitched_doc, chosen_doc[1], chosen_doc[2]  # Return stitched doc with random Q&A

    def get_answer(self, answer: str) -> Answer:
        return Answer(
            answer=answer,
            start=0,
            end=0
        )


if __name__ == '__main__':
    data_handler = StitchedTechQADataHandler(minimum_context_characters=50000)
    data = data_handler.load_data(limit=10)
    print(data)
