from tqdm.auto import tqdm
import json

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
            return load_from_disk(path).to_list()
