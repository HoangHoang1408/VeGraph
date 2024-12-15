import os
from utils.load_data import LoadData

class DatasetLoader:
    def __init__(self, processed_data_path):
        self.datasets = {
            "hover": {
                "2hop": LoadData.load_data(os.path.join(processed_data_path, "hover", "2hop.json")),
                "3hop": LoadData.load_data(os.path.join(processed_data_path, "hover", "3hop.json")),
                "4hop": LoadData.load_data(os.path.join(processed_data_path, "hover", "4hop.json")),
            },
            "feverous": {
                "feverous_sentence": LoadData.load_data(os.path.join(processed_data_path, "feverous", "feverous_sentence.json"))
            }
        }

    def get_partition(self, dataset_name, dataset_partition):
        return self.datasets[dataset_name][dataset_partition]