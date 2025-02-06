from abc import ABC
from abc import abstractmethod
from typing import Dict, List

from src.config.config import JsonReaderConfig
from src.json_reader.json_reader import JsonReader
from src.utils import mean_of_lists


class Stats(ABC):

    @abstractmethod
    def get_scores(self, json_data: List[Dict]): ...


class RecallStats(Stats):

    def get_scores(self, json_data: List[Dict]):
        result = {}
        for key, chunker_results in json_data.items():
            result[key] = self.recall_mean([i["recall_at_k"] for i in chunker_results if len(i["recall_at_k"]) == 5])
        return result


    def recall_mean(self, list_of_recalls: List[List[float]]) -> List[float]:
        return mean_of_lists(list_of_recalls)


class MeanStats(Stats):

    def get_scores(self, json_data: List[Dict]):
        return self.get_mean_scores(json_data)

    def update_moving_average(self, old_avg, new_value, count):
        """
        Calculates the moving average of a series of values.
        It can be useful not to have to keep many values in memory.
        """
        return old_avg + ((new_value - old_avg) / count)

    def get_k(self, eval_results) -> int:
        return len(eval_results[0]["recall_at_k"])

    def get_mean_scores(self, json_data) -> Dict:
        mean_scores = {}
        for file_path, eval_results in json_data.items():
            k = self.get_k(eval_results)
            mean_scores[file_path] = {
                str(num): 0 for num in range(1, k + 1)
            }
            for doc_num, document in enumerate(eval_results, start=1):
                for retriever_result in document['retriever_results']:
                    scores = retriever_result['scores']
                    for score_num, score in enumerate(scores, start=1):
                        old_avg = mean_scores[file_path][str(score_num)]
                        new_avg = self.update_moving_average(old_avg, score, doc_num)
                        mean_scores[file_path][str(score_num)] = new_avg

            mean_scores[file_path] = {k: round(v, 3) for k, v in mean_scores[file_path].items()}
        return mean_scores


if __name__ == '__main__':
    config = JsonReaderConfig()
    reader = JsonReader(config)
    json_data = reader.read_jsons_from_directory()

    stats = RecallStats()
    scores = stats.get_scores(json_data)
    for key, value in scores.items():
        print(f"{key}: {value}")
