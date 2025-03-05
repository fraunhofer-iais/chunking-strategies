import json
import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt


def read_dataset_json(filename: str) -> List[Dict]:
    with open (filename, "r") as f:
        dataset = json.load(f)

    return dataset


def get_avg_document_length(dataset: List[Dict]) -> float:
    total_length = 0
    for sample in dataset:
        total_length += len(sample["document"])
    return total_length / len(dataset)


def get_avg_question_length(dataset: List[Dict]) -> float:
    total_length = 0
    for sample in dataset:
        for question in sample["questions"]:
            total_length += len(question)
    return total_length / len(dataset)


def get_avg_answer_length(dataset: List[Dict]) -> float:
    total_length = 0
    for sample in dataset:
        for answer_data in sample["answers"]:
            answer = answer_data["answer"]
            total_length += len(answer)
    return total_length / len(dataset)


def get_avg_num_of_questions(dataset: List[Dict]) -> float:
    total_questions = 0
    for sample in dataset:
        total_questions += len(sample["questions"])
    return total_questions / len(dataset)


def get_num_documents(dataset: List[Dict]) -> int:
    return len(dataset)


def tokenize(text):
    token_pattern = r"""
        \b(?:[A-Z]\.)+[A-Z]?       # Abbreviations like U.S.A. or U.K.
        | \b\w+(?:[-']\w+)*\b      # Words, including contractions and hyphenated words
        | \.\.\.                   # Ellipses (...)
        | [.,!?;(){}\[\]:\"“”]     # Punctuation
        | (?:https?://|www\.)\S+   # URLs
        | [#@]\w+                  # Hashtags and mentions
        | \b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b  # Numbers with commas or decimals
        | [\U0001F600-\U0001F64F]  # Emojis (basic range)
        | \S                        # Any other non-whitespace character
    """
    
    return re.findall(token_pattern, text, re.VERBOSE)

def get_avg_num_tokens(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        total_tokens += len(tokenize(sample["document"]))
    return total_tokens / len(dataset)

def get_avg_num_tokens_questions(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        for question in sample["questions"]:
            total_tokens += len(tokenize(question))
    return total_tokens / len(dataset)

def get_avg_num_tokens_answers(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        for answer_data in sample["answers"]:
            answer = answer_data["answer"]
            total_tokens += len(tokenize(answer))
    return total_tokens / len(dataset)


def tokenize_by_whitespace(text):
    return text.split()


def get_avg_num_tokens_by_whitespace(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        total_tokens += len(tokenize_by_whitespace(sample["document"]))
    return total_tokens / len(dataset)


def get_avg_num_tokens_questions_by_whitespace(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        for question in sample["questions"]:
            total_tokens += len(tokenize_by_whitespace(question))
    return total_tokens / len(dataset)


def get_avg_num_tokens_answers_by_whitespace(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        for answer_data in sample["answers"]:
            answer = answer_data["answer"]
            total_tokens += len(tokenize_by_whitespace(answer))
    return total_tokens / len(dataset)


def get_avg_num_unique_tokens(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        total_tokens += len(set(tokenize(sample["document"])))
    return total_tokens / len(dataset)


def get_avg_num_unique_tokens_by_whitespace(dataset: List[Dict]) -> float:
    total_tokens = 0
    for sample in dataset:
        total_tokens += len(set(tokenize_by_whitespace(sample["document"])))
    return total_tokens / len(dataset)

# TODO: Write a function that stores all outcome values in a dictionary.
def get_all_stats(dataset: List[Dict]) -> Dict:
    stats = {
        "avg_doc_length": get_avg_document_length(dataset),
        "avg_question_length": get_avg_question_length(dataset),
        "avg_answer_length": get_avg_answer_length(dataset),
        "avg_num_questions": get_avg_num_of_questions(dataset),
        "num_documents": get_num_documents(dataset),
        "avg_num_tokens": get_avg_num_tokens(dataset),
        "avg_num_tokens_questions": get_avg_num_tokens_questions(dataset),
        "avg_num_tokens_answers": get_avg_num_tokens_answers(dataset),
        "avg_num_tokens_by_whitespace": get_avg_num_tokens_by_whitespace(dataset),
        "avg_num_tokens_questions_by_whitespace": get_avg_num_tokens_questions_by_whitespace(dataset),
        "avg_num_tokens_answers_by_whitespace": get_avg_num_tokens_answers_by_whitespace(dataset),
        "avg_num_unique_tokens": get_avg_num_unique_tokens(dataset),
        "avg_num_unique_tokens_by_whitespace": get_avg_num_unique_tokens_by_whitespace(dataset)
    }
    return stats

def get_all_stats_rounded_to_1(dataset: List[Dict]) -> Dict:
    stats = get_all_stats(dataset)
    for key in stats:
        stats[key] = round(stats[key], 1)
    return stats


def write_stats(stats: Dict, filename: str) -> None:
    outf = filename.replace(".json", "_stats.json")
    with open (outf, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

def plot_distribution_of_document_lengths(filename: str, dataset: List[Dict]) -> None:
    document_lengths = [len(sample["document"]) for sample in dataset]
    plt.hist(document_lengths, bins=50)
    plt.xlabel("Document Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Document Lengths")
    plot_name = filename.replace(".json", "_document_lengths.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_question_lengths(filename: str, dataset: List[Dict]) -> None:
    question_lengths = [len(question) for sample in dataset for question in sample["questions"]]
    plt.hist(question_lengths, bins=50)
    plt.xlabel("Question Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Question Lengths")
    plot_name = filename.replace(".json", "_question_lengths.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_answer_lengths(filename: str, dataset: List[Dict]) -> None:
    answer_lengths = [len(answer_data["answer"]) for sample in dataset for answer_data in sample["answers"]]
    plt.hist(answer_lengths, bins=50)
    plt.xlabel("Answer Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Answer Lengths")
    plot_name = filename.replace(".json", "_answer_lengths.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_num_questions(filename: str, dataset: List[Dict]) -> None:
    num_questions = [len(sample["questions"]) for sample in dataset]
    plt.hist(num_questions, bins=50)
    plt.xlabel("Number of Questions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Questions")
    plot_name = filename.replace(".json", "_num_questions.png")
    plt.savefig(plot_name)
    plt.clf()


def plot_distribution_of_num_tokens(filename: str, dataset: List[Dict]) -> None:
    num_tokens = [len(tokenize(sample["document"])) for sample in dataset]
    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens")
    plot_name = filename.replace(".json", "_num_tokens.png")
    plt.savefig(plot_name)
    plt.clf()


def plot_distribution_of_num_tokens_by_whitespace(filename: str, dataset: List[Dict]) -> None:
    num_tokens = [len(tokenize_by_whitespace(sample["document"])) for sample in dataset]
    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens by Whitespace")
    plot_name = filename.replace(".json", "_num_tokens_whitespace.png")
    plt.savefig(plot_name)
    plt.clf()


def plot_distribution_of_num_tokens_questions(filename: str, dataset: List[Dict]) -> None:
    num_tokens = [len(tokenize(question)) for sample in dataset for question in sample["questions"]]
    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens in Questions")
    plot_name = filename.replace(".json", "_num_tokens_questions.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_num_tokens_questions_by_whitespace(filename: str, dataset: List[Dict]) -> None:
    num_tokens = []
    for sample in dataset:
        for question in sample["questions"]:
            tokenized = tokenize_by_whitespace(question)
            num_tokens.append(len(tokenized))

    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens in Questions by Whitespace")
    plot_name = filename.replace(".json", "_num_tokens_questions_whitespace.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_num_tokens_answers(filename: str, dataset: List[Dict]) -> None:
    num_tokens = []
    for sample in dataset:
        for answer_data in sample["answers"]:
            tokenized = tokenize(answer_data["answer"])
            num_tokens.append(len(tokenized))

    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens in Answers")
    plot_name = filename.replace(".json", "_num_tokens_answers.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_num_tokens_answers_by_whitespace(filename: str, dataset: List[Dict]) -> None:
    num_tokens = []
    for sample in dataset:
        for answer_data in sample["answers"]:
            tokenized = tokenize_by_whitespace(answer_data["answer"])
            num_tokens.append(len(tokenized))

    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens in Answers by Whitespace")
    plot_name = filename.replace(".json", "_num_tokens_answers_whitespace.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_num_unique_tokens(filename: str, dataset: List[Dict]) -> None:
    num_tokens = []
    for sample in dataset:
        tokenized = tokenize(sample["document"])
        num_tokens.append(len(tokenized))

    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Unique Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Unique Tokens")
    plot_name = filename.replace(".json", "_num_unique_tokens.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_distribution_of_num_unique_tokens_by_whitespace(filename: str, dataset: List[Dict]) -> None:
    num_tokens = []
    for sample in dataset:
        tokenized = tokenize_by_whitespace(sample["document"])
        num_tokens.append(len(tokenized))

    plt.hist(num_tokens, bins=50)
    plt.xlabel("Number of Unique Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Unique Tokens by Whitespace")
    plot_name = filename.replace(".json", "_num_unique_tokens_whitespace.png")
    plt.savefig(plot_name)
    plt.clf()

def plot_all_distributions(filename: str, dataset: List[Dict]) -> None:
    plot_distribution_of_document_lengths(filename, dataset)
    plot_distribution_of_question_lengths(filename, dataset)
    plot_distribution_of_answer_lengths(filename, dataset)
    plot_distribution_of_num_questions(filename, dataset)
    plot_distribution_of_num_tokens(filename, dataset)
    plot_distribution_of_num_tokens_by_whitespace(filename, dataset)
    plot_distribution_of_num_tokens_questions(filename, dataset)
    plot_distribution_of_num_tokens_questions_by_whitespace(filename, dataset)
    plot_distribution_of_num_tokens_answers(filename, dataset)
    plot_distribution_of_num_tokens_answers_by_whitespace(filename, dataset)
    plot_distribution_of_num_unique_tokens(filename, dataset)
    plot_distribution_of_num_unique_tokens_by_whitespace(filename, dataset)


if __name__ == "__main__":
    filename = sys.argv[1]
    dataset = read_dataset_json(filename)
    stats = get_all_stats_rounded_to_1(dataset)
    plot_all_distributions(filename, dataset)
    write_stats(stats, filename)
