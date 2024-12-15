import json
import os
import random
from argparse import ArgumentParser
from tqdm.auto import tqdm


class LoadData:
    @staticmethod
    def load_data(path):
        if path.endswith(".jsonl"):
            data = []
            with open(path) as f:
                for l in tqdm(f.readlines()):
                    data.append(json.loads(l))
                return data
        elif path.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        elif path.endswith(".txt"):
            with open(path) as f:
                return f.read()
        else:
            raise Exception("Invalid data file")

class DatasetLoader:
    def __init__(self, dataset_folder):
        def is_valid_sample(evidence):
            for content in evidence:
                for element in content['content']:
                    if element.find('_sentence_') < 0:
                        return False
            return True
        
        hover_train = LoadData.load_data(os.path.join(dataset_folder, "hover/hover_train_release_v1.1.json"))
        hover_dev = LoadData.load_data(os.path.join(dataset_folder, "hover/hover_dev_release_v1.1.json"))
        feverous_train = LoadData.load_data(os.path.join(dataset_folder, "feverous/feverous_train_challenges.jsonl"))
        feverous_train = [{k: v for k, v in x.items() if k in ['id', 'claim', 'label', 'challenge']} for x in feverous_train if is_valid_sample(x['evidence']) and x['label'] in set(['SUPPORTS', 'REFUTES'])]
        feverous_dev = LoadData.load_data(os.path.join(dataset_folder, "feverous/feverous_dev_challenges.jsonl"))
        feverous_dev = [{k: v for k, v in x.items() if k in ['id', 'claim', 'label', 'challenge']} for x in feverous_dev if is_valid_sample(x['evidence']) and x['label'] in set(['SUPPORTS', 'REFUTES'])]

        self.dataset = {
            "hover": {
                "train": {
                    "2hop": [x for x in hover_train if x["num_hops"] == 2],
                    "3hop": [x for x in hover_train if x["num_hops"] == 3],
                    "4hop": [x for x in hover_train if x["num_hops"] == 4],
                },
                "dev": {
                    "2hop": [x for x in hover_dev if x["num_hops"] == 2],
                    "3hop": [x for x in hover_dev if x["num_hops"] == 3],
                    "4hop": [x for x in hover_dev if x["num_hops"] == 4],
                },
            },
            "feverous": {
                "train": {
                    "combining_tables_and_text": [x for x in feverous_train if x["challenge"] == "Combining Tables and Text"],
                    "entity_disambiguation": [x for x in feverous_train if x["challenge"] == "Entity Disambiguation"],
                    "multi_hop_reasoning": [x for x in feverous_train if x["challenge"] == "Multi-hop Reasoning"],
                    "numerical_reasoning": [x for x in feverous_train if x["challenge"] == "Numerical Reasoning"],
                    "other": [x for x in feverous_train if x["challenge"] == "Other"],
                    "search_terms_not_in_claim": [x for x in feverous_train if x["challenge"] == "Search terms not in claim"],
                },
                "dev": {
                    "combining_tables_and_text": [x for x in feverous_dev if x["challenge"] == "Combining Tables and Text"],
                    "entity_disambiguation": [x for x in feverous_dev if x["challenge"] == "Entity Disambiguation"],
                    "multi_hop_reasoning": [x for x in feverous_dev if x["challenge"] == "Multi-hop Reasoning"],
                    "numerical_reasoning": [x for x in feverous_dev if x["challenge"] == "Numerical Reasoning"],
                    "other": [x for x in feverous_dev if x["challenge"] == "Other"],
                    "search_terms_not_in_claim": [x for x in feverous_dev if x["challenge"] == "Search terms not in claim"],
                },
            },
        }

    def get_partition(self, dataset_name, dataset_split, dataset_partition):
        label_mapper = {
            "SUPPORTED": True,
            "NOT_SUPPORTED": False,
            "SUPPORTS": True,
            "REFUTES": False
        }
        partition = []
        for sample in self.dataset[dataset_name][dataset_split][dataset_partition]:
            id_ = None
            if dataset_name == "hover":
                id_ = sample["uid"]
            elif dataset_name == "feverous":
                id_ = sample["id"]
    
            partition.append({
                "id": id_,
                "claim": sample["claim"],
                "label": label_mapper[sample["label"]],
            })
        return partition

    def sample_from_partition(self, partition_samples, seed=148, n_samples=200):
        random.seed(seed)
        random.shuffle(partition_samples)
        n_samples = int(n_samples / 2)
        true_samples = [x for x in partition_samples if x['label'] == True][:n_samples]
        false_samples = [x for x in partition_samples if x['label'] == False][:n_samples]
        samples = true_samples + false_samples
        random.shuffle(samples)
        return samples

    def process_data(self, seed=148, n_samples=200, save_folder="."):
        data = {
            "hover": {
                "2hop": self.sample_from_partition(self.get_partition("hover", "dev", "2hop"), seed=seed, n_samples=n_samples),
                "3hop": self.sample_from_partition(self.get_partition("hover", "dev", "3hop"), seed=seed, n_samples=n_samples),
                "4hop": self.sample_from_partition(self.get_partition("hover", "dev", "4hop"), seed=seed, n_samples=n_samples),
            },
            "feverous": {
                "entity_disambiguation": self.get_partition("feverous", "dev", "entity_disambiguation"),
                "multi_hop_reasoning": self.get_partition("feverous", "dev", "multi_hop_reasoning"),
                "numerical_reasoning": self.get_partition("feverous", "dev", "numerical_reasoning"),
                "search_terms_not_in_claim": self.get_partition("feverous", "dev", "search_terms_not_in_claim"),
                "other": self.get_partition("feverous", "dev", "other"),
            }
        }
        
        dataset = "hover"
        partitions = data['hover']
        os.makedirs(os.path.join(save_folder, dataset), exist_ok=True)
        for partition_name, partition_data in partitions.items():
            partition_path = os.path.join(save_folder, dataset, f"{partition_name}.json")
            with open(partition_path, "w") as f:
                json.dump(partition_data, f)

        dataset = "feverous"
        partition_name = "feverous_sentence"
        os.makedirs(os.path.join(save_folder, dataset), exist_ok=True)
        partition_data = data['feverous']['entity_disambiguation'] + data['feverous']['multi_hop_reasoning'] + data['feverous']['numerical_reasoning'] + data['feverous']['search_terms_not_in_claim']
        random.seed(seed)
        partition_data = partition_data + random.sample(data['feverous']['other'], k=(800 - len(partition_data)))
        partition_path = os.path.join(save_folder, dataset, f"{partition_name}.json")
        with open(partition_path, "w") as f:
            json.dump(partition_data, f)
        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-folder", type=str)
    parser.add_argument("--processed-data-folder", type=str)
    parser.add_argument("--hover-partition-n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=148)
    args = parser.parse_args()

    dataset_loader = DatasetLoader(args.dataset_folder)
    dataset_loader.process_data(args.seed, args.hover_partition_n_samples, args.processed_data_folder)