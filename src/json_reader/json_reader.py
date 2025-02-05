import json
import os

from src.config.config import JsonReaderConfig


class JsonReader:

    def __init__(self, config: JsonReaderConfig):
        self._config = config

    def read_jsons_from_directory(self):
        """Reads all JSON files in a given directory and returns a list of dictionaries.

        Args:
          directory: The path to the directory containing the JSON files.

        Returns:
          A dictionary of JSON filenames mapped to a list of evaluation samples.
        """
        json_files = [f for f in os.listdir(self._config.json_dir) if f.endswith('.json') and "configs.json" not in f and "scores.json" not in f]
        json_data = {}
        for file in json_files:
            file_path = os.path.join(self._config.json_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                json_data[file_path] = data
        return json_data
