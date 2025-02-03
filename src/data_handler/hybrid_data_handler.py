import random
from typing import List, Optional

from src.data_handler.data_handler import DataHandler
from src.data_handler.stitched_squad_data_handler import StitchedSquadDataHandler
from src.data_handler.stitched_tech_qa_data_handler import StitchedTechQADataHandler
from src.dto.dto import EvalSample, Answer

class HybridDataHandler(DataHandler):
    def __init__(self, handlers: List[DataHandler], limit_samples_per_dataset: int):
        self.handlers = handlers
        self.limit_samples_per_dataset = limit_samples_per_dataset # TODO: discuss how to handle limit and limit_samples_per_dataset

    def load_data(self, limit: Optional[int] = None) -> List[EvalSample]:
        all_samples = []
        for handler in self.handlers:
            samples = handler.load_data(self.limit_samples_per_dataset)
            all_samples.append(samples)
        return self._create_hybrid_samples(all_samples, limit)

    def _create_hybrid_samples(self, all_samples: List[List[EvalSample]], limit: Optional[int]) -> List[EvalSample]:
        hybrid_samples = []
        num_samples = min(len(samples) for samples in all_samples)
        if limit is not None:
            num_samples = min(num_samples, limit)

        for i in range(num_samples):
            stitched_document = ""
            all_questions = []
            all_answers = []

            for dataset_samples in all_samples:
                sample = dataset_samples[i]
                stitched_document += " " + sample.document
                all_questions.extend(sample.questions)
                all_answers.extend(sample.answers)

            stitched_document = stitched_document.strip()

            # Randomly select a question-answer pair from the same sample
            if all_questions and all_answers:
                random_index = random.randint(0, len(all_questions) - 1)
                chosen_question = all_questions[random_index]
                chosen_answer = all_answers[random_index]

                hybrid_samples.append(
                    EvalSample(
                        document_id=f"hybrid_{i+1}",
                        document=stitched_document,
                        questions=[chosen_question],
                        answers=[Answer(answer=chosen_answer.answer, start=chosen_answer.start, end=chosen_answer.end)],
                    )
                )

        return hybrid_samples

if __name__ == '__main__':
    handler1 = StitchedSquadDataHandler(minimum_context_characters=50000)
    handler2 = StitchedTechQADataHandler(minimum_context_characters=50000)

    handlers = [handler1, handler2]
    hybrid_handler = HybridDataHandler(handlers, limit_samples_per_dataset =2)
    hybrid_data = hybrid_handler.load_data()
    print(hybrid_data)
