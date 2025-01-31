from src.config.config import JsonReaderConfig
from src.evaluator.json_reader import JsonReader


def update_moving_average(old_avg, new_value, count):
    """
    Calculates the moving average of a series of values.
    It can be useful not to have to keep many values in memory.
    """
    return old_avg + ((new_value - old_avg) / count)


def get_k(eval_results):
    return len(eval_results[0]["recall_at_k"])


def get_mean_scores(json_data):
    mean_scores = {}
    for file_path, eval_results in json_data.items():
        k = get_k(eval_results)
        mean_scores[file_path] = {
            str(num): 0 for num in range(1, k + 1)
        }
        for doc_num, document in enumerate(eval_results, start=1):
            for retriever_result in document['retriever_results']:
                scores = retriever_result['scores']
                for score_num, score in enumerate(scores, start=1):
                    old_avg = mean_scores[file_path][str(score_num)]
                    new_avg = update_moving_average(old_avg, score, doc_num)
                    mean_scores[file_path][str(score_num)] = new_avg

        mean_scores[file_path] = {k: round(v, 3) for k, v in mean_scores[file_path].items()}
    return mean_scores


if __name__ == '__main__':
    config = JsonReaderConfig()
    reader = JsonReader(config)
    json_data = reader.read_jsons_from_directory()

    mean_scores = get_mean_scores(json_data)
    print(mean_scores)
