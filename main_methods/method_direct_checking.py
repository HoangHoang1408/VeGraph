import os
from argparse import ArgumentParser
from functools import partial
from pprint import pprint
from uuid import uuid4
from datetime import datetime
import json
import requests
from dataset_loader import DatasetLoader
from utils.get_completion import get_completion
from utils.json_parser import JSONParser
from utils.multi_process_task_dict import multi_process_task_dict

DIRECT_PROMPT = """
### Task: You will be given a claim, you job is to verify the veracity of the claim base on your knowledge. Return in the following JSON format
{
    "rationale": "your thought on the veracity of the claim",
    "veracity": "true or false boolean value"
}

### Input claim: {{claim}}
""".strip()

def verify_claim(sample, get_completion, n_results=1):
    prompt = DIRECT_PROMPT.replace("{{claim}}", sample['claim'])
    sample['predicted_results'] = []
    for _ in range(n_results):
        res = get_completion(prompt)
        try:
            veracity = JSONParser.extract_json_dict(res)
            if not isinstance(veracity['veracity'], bool):
                raise Exception()
        except:
            veracity = {
                "rationale": None,
                "veracity": False
            }
            for term in ['true', 'True', 'TRUE']:
                if term in res:
                    veracity['veracity'] = True
                    break
        sample['predicted_results'].append(veracity)
    return sample

def get_current_date_time():
    return datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--llm-host-address", type=str, default="http://10.254.138.192:9002")
    parser.add_argument("--job-version", type=str, default=get_current_date_time())
    parser.add_argument("--job-folder", type=str)
    parser.add_argument("--preprocessed-dataset-path", type=str)
    parser.add_argument("--dataset", type=str, choices=['hover', 'feverous'])
    parser.add_argument("--partition", type=str)
    
    parser.add_argument("--cache-completion", action="store_true")
    parser.add_argument("--use-cached-completion", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-results", type=int, default=1)

    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = parse_args()
    get_completion = partial(get_completion, temperature=args.temperature, use_cached_result=args.use_cached_completion, cache_result=args.cache_completion, host=args.llm_host_address)
    args.model = requests.get(f"{args.llm_host_address}/v1/models").json()['data'][0]['id']
    pprint(vars(args))

    # prepare benchmark folder
    job_id = uuid4().hex
    data_loader = DatasetLoader(args.preprocessed_dataset_path)
    partition = data_loader.get_partition(dataset_name=args.dataset, dataset_partition=args.partition)
    
    task_dict = {sample['id']: lambda sample=sample, get_completion=get_completion, n_results=args.n_results: verify_claim(sample, get_completion, n_results) for sample in partition}
    results = multi_process_task_dict(task_dict, num_workers=args.n_workers)

    save_folder = os.path.join(args.job_folder, job_id)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "arguments.json"), "w") as f:
        json.dump(vars(args), f)
    with open(os.path.join(save_folder, "results.json"), "w") as f:
        json.dump(list(results.values()), f, ensure_ascii=False)
    
        












