from typing import List, Optional

from src.config.config import HybridDataHandlerConfig
from src.data_handler.data_handler import DataHandler
from src.data_handler.stitched_squad_data_handler import StitchedSquadDataHandler
from src.data_handler.stitched_tech_qa_data_handler import StitchedTechQADataHandler
from src.dto.dto import EvalSample


class HybridDataHandler(DataHandler):
    def __init__(self, config: HybridDataHandlerConfig):
        self.config = config
        self.limit_samples_per_dataset = self.config.limit_samples_per_dataset

    def load_data(self, limit: Optional[int] = None) -> List[EvalSample]:
        all_samples = []
        from src.factory.data_handler_factory import DataHandlerFactory
        for handler_config in self.config.handler_configs:
            from src.factory.data_handler_factory import DataHandlerFactory
            data_handler = DataHandlerFactory.create(handler_config)
            samples = data_handler.load_data(limit=self.limit_samples_per_dataset)
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

            valid_questions = []
            valid_answers = []

            for question, answer in zip(all_questions, all_answers):
                if stitched_document.lower().count(answer.answer.lower()) == 1:
                    valid_questions.append(question)
                    valid_answers.append(answer)

            if valid_questions and valid_answers:
                hybrid_samples.append(
                    EvalSample(
                        document_id=f"hybrid_{i + 1}",
                        document=stitched_document,
                        questions=valid_questions,
                        answers=valid_answers,
                    )
                )

        return hybrid_samples


if __name__ == '__main__':
    handler1_config = StitchedSquadDataHandler(minimum_context_characters=50000)
    handler2_config = StitchedTechQADataHandler(minimum_context_characters=50000)
    hybrid_config = HybridDataHandlerConfig(
        handler_configs=[handler1_config, handler2_config],
        limit_samples_per_dataset=2
    )
    hybrid_handler = HybridDataHandler(hybrid_config)
    hybrid_data = hybrid_handler.load_data(limit=5)
    print(hybrid_data)
