import argparse

from src.data_handler.hotpot_qa_data_handler import HotpotQADataHandler
from src.data_handler.narrative_qa_data_handler import NarrativeQADataHandler

dataset_factory = {
    "TIGER-Lab/LongRAG": HotpotQADataHandler,
    "deepmind/narrativeqa": NarrativeQADataHandler,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d", type=str, default="TIGER-Lab/LongRAG")
    parser.add_argument("--max_samples", "-m", type=int, default=5)

    args = parser.parse_args()
    
    dataset_class = dataset_factory[args.dataset_name]
    data_handler = dataset_class()
    data = data_handler.load_data()

    for num, sample in enumerate(data):
        if num > args.max_samples:
            break

        spans = sample.answers[0].spans
        for span_num, span in enumerate(spans):
            question = sample.questions[0]
            anwer = sample.answers[0].answer
            document = sample.document
            answer_span = document[span.start:span.end]

            print(f"Sample {num}")
            print(f"Span {span_num}")
            print(f"Question: {question}")
            print(f"Answer: {anwer}")
            print()
            print(f"Answer: {answer_span}")
        print()


if __name__ == "__main__":
    main()
