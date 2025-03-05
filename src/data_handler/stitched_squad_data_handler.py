import random
from typing import List, Set, Tuple

from tqdm import tqdm

from src.data_handler.data_handler import DataHandler
from src.dto.dto import EvalSample, Answer


class StitchedSquadDataHandler(DataHandler):

    dataset_name: str = "lhoestq/squad"

    def __init__(self, minimum_context_characters: int):
        self.minimum_context_characters = minimum_context_characters

    def _extract_documents(self, dataset, document_id: int, seen_documents: Set[str], limit: int, pbar: tqdm) \
            -> List[EvalSample]:
        buffer = []  # Store short documents for stitching
        samples = []

        for dataset_sample in enumerate(dataset):
            if limit and len(samples) >= limit:
                break
            pbar.update(1)
            document = dataset_sample[1]["context"]
            question = dataset_sample[1]["question"]
            answer_data = dataset_sample[1]["answers"]
            answer = self._get_answer(answer_data)

            # Skip duplicate documents
            if document in seen_documents:
                continue
            seen_documents.add(document)

            # If document is long enough, use it directly
            if len(document) >= self.minimum_context_characters:
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
                    stitched_doc, valid_qas = self._stitch_documents(buffer)
                    buffer = []  # Reset buffer after stitching

                    if valid_qas:  # Ensure there are valid Q&A pairs
                        questions, answers = zip(*valid_qas)
                        sample = EvalSample(
                            document_id=f"stitched_{document_id}",
                            document=stitched_doc,
                            questions=list(questions),
                            answers=list(answers),
                        )
                        samples.append(sample)
                        document_id += 1

        return samples

    def _stitch_documents(self, buffer: List[Tuple[str, str, Answer]]) -> Tuple[str, List[Tuple[str, Answer]]]:
        """ Stitches multiple short documents into one and selects a random Q&A pair """
        random.shuffle(buffer)  # Shuffle to mix documents
        stitched_doc = " ".join([doc[0] for doc in buffer])  # Combine all texts
        valid_qas = []
        for doc, question, answer in buffer:
            # Check if the answer appears exactly once in the stitched document
            if stitched_doc.lower().count(answer.answer.lower()) == 1:
                valid_qas.append((question, answer))

        return stitched_doc, valid_qas  # Return stitched doc with all questions and answers

    def _get_answer(self, answer: dict) -> Answer:
        """ Extracts the first answer span from the dataset """
        return Answer(
            answer=answer["text"][0],
            start=answer["answer_start"][0],
            end=answer["answer_start"][0] + len(answer["text"][0])
        )


if __name__ == '__main__':
    data_handler = StitchedSquadDataHandler(minimum_context_characters=50000)
    data = data_handler.load_data(limit=5)
    print(data)
