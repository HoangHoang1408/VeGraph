import json
import os
import shutil
import traceback
from argparse import ArgumentParser
from copy import copy, deepcopy
from datetime import datetime
from functools import partial
from pprint import pprint
from uuid import uuid4

import requests
from dataset_loader import DatasetLoader
from prompt_loader import PromptLoader
from qa import QA
from retriever import Retriever
from utils.get_completion import get_completion
from utils.json_parser import JSONParser
from utils.multi_process_task_dict import multi_process_task_dict


class StepParser:
    @staticmethod
    def parse_clarified_claim(response):
        return " ".join(JSONParser.extract_json_dict(response)['clarified_claim'].split("\n"))

    @staticmethod
    def parse_graph(response):
        guidance, graph = response.split("<guidance_for_graph_construction>")[1].split("<graph>")
        return {
            "guidance": guidance.strip(),
            "graph": graph.strip()
        }

    @staticmethod
    def group_hidden_entity(graph_text):
        hidden_entities = set()
        for triplet in graph_text.split("\n"):
            for element in triplet.split("||"):
                if "hidden" in element:
                    hidden_entities.add(element)
        hidden_triplets = {e: [] for e in hidden_entities}
        for element in hidden_entities:
            for triplet in graph_text.split("\n"):
                if len([x for x in triplet.split("||") if 'hidden' in x]) > 1:
                    continue
                if element in triplet:
                    hidden_triplets[element].append(triplet)
        hidden_triplets = {k: {"hidden_triplets_text": "\n".join(v)} for k, v in hidden_triplets.items() if len(v) > 0}
        hidden_triplets = dict(sorted(hidden_triplets.items()))
        return hidden_triplets

    @staticmethod
    def parse_query(response):
        return JSONParser.extract_json_dict(response)


class ZeroShotFactCheckingAgent:
    def __init__(self, claim):
        self.retriever = Retriever()
        self.get_completion = partial(get_completion, temperature=args.temperature, use_cached_result=args.use_cached_completion, cache_result=args.cache_completion, host=args.llm_host_address)
        self.qa = QA(self.get_completion)
        self.prompt_loader = PromptLoader()
        self.state = {"claim": claim}
        self.parser = StepParser

    def _claim_clarify(self):
        # prompt = self.prompt_loader.construct_zero_shot_claim_clarify_prompt(self.state["claim"])
        # res = self.get_completion(prompt)
        # self.state["clarified_claim"] = StepParser.parse_clarified_claim(res)

        # test perf, using 
        self.state["clarified_claim"] = self.state["claim"]

    def _construct_graph(self):
        if args.construct_graph_examples_set_id is not None:
            prompt = self.prompt_loader.construct_few_shot_construct_graph_prompt(self.state["clarified_claim"], args.dataset, args.construct_graph_examples_prompt_set_path, args.construct_graph_examples_set_id)
        else:
            prompt = self.prompt_loader.construct_zero_shot_construct_graph_prompt(self.state["clarified_claim"])
        res = self.get_completion(prompt)
        res = StepParser.parse_graph(res)
        self.state["graph"] = res["graph"]
        self.state["guidance_for_graph_construction"] = res["guidance"]

    def _get_prev_query_gen_failed_trails(self):
        query_gen_failed_trails = {}
        resolve_step = len(self.state['resolve_entity_steps']) - 1
        for step in range(resolve_step):
            prev_step_hidden_triplets = self.state['resolve_entity_steps'][step]['hidden_triplets']
            current_step_hidden_triplets = self.state["resolve_entity_steps"][-1]["hidden_triplets"]
            for key in current_step_hidden_triplets:
                if key in prev_step_hidden_triplets and prev_step_hidden_triplets[key]['resolved_entity'] is None:
                    if key not in query_gen_failed_trails:
                        query_gen_failed_trails[key] = []
                    query_gen_failed_trails[key].append({
                        "question": self.state['resolve_entity_steps'][step]['hidden_triplets'][key]['question'],
                        "rationale": self.state['resolve_entity_steps'][step]['hidden_triplets'][key]['rationale']
                    })
        return query_gen_failed_trails
                    
    def _resolve_entity(self):
        self.state["resolve_entity_steps"] = []
        # refinement loop
        for resolve_step in range(args.resolve_entity_max_loop):
            if resolve_step == 0:
                step_state = {"graph": copy(self.state["graph"])}
            else:
                step_state = {"graph": copy(self.state["resolve_entity_steps"][resolve_step - 1]['resolved_graph'])}
            self.state["resolve_entity_steps"].append(step_state)
            step_state['resolved_graph'] = copy(step_state['graph'])
            
            # get all hidden entities and their corresponding triplets
            hidden_triplets = StepParser.group_hidden_entity(step_state["graph"])
            if len(hidden_triplets) == 0:
                break
            step_state["hidden_triplets"] = hidden_triplets

            # get all the previous failed trails
            query_gen_failed_trails = self._get_prev_query_gen_failed_trails()
            
            # resolve entities gradually
            for entity_key, entity_value in step_state["hidden_triplets"].items():
                # construct prompt
                entity_triplet = entity_value['hidden_triplets_text'].split("\n")
                entity_graph = "\n".join(f"{i}||{e}" for i, e in enumerate(entity_triplet)).replace(entity_key, "hidden_entity")
                if entity_key not in query_gen_failed_trails:
                    prompt = self.prompt_loader.construct_zero_shot_query_gen_prompt(claim=self.state['clarified_claim'], graph=entity_graph)
                else:
                    failed_trails = query_gen_failed_trails[entity_key]
                    prompt = self.prompt_loader.construct_zero_shot_query_refine_prompt(claim=self.state['clarified_claim'], graph=entity_graph, failed_trails=failed_trails)

                # get_completion and parse question
                res = self.get_completion(prompt)
                entity_value.update({k: v for k, v in JSONParser.extract_json_dict(res).items() if k in ['rationale', 'question', 'triplet_ids']})
                entity_value['resolved_entity'] = self.qa.answer_question(entity_value['question'], corpus_name=args.dataset, use_rerank=args.use_rerank)['answer']

                # update the entity in the resolved graph
                if entity_value['resolved_entity'] is not None:
                    step_state['resolved_graph'] = step_state['resolved_graph'].replace(entity_key, str(entity_value['resolved_entity']))

    def _check_pass_resolve_entity(self):
        final_resolved_graph = self.state['resolve_entity_steps'][-1]['resolved_graph']
        if "hidden_entity" in final_resolved_graph:
            return False
        return True
    
    def _check_subclaim(self):
        # get already checked triplets
        valid_subclaims = {}
        for step in self.state['resolve_entity_steps'][:-1]:
            for hidden_entity, hidden_triplets in step['hidden_triplets'].items():
                if hidden_triplets['resolved_entity'] is None: continue
                hidden_triplets_text = hidden_triplets["hidden_triplets_text"].split("\n")
                valid_subclaims[hidden_entity] = {
                    "triplets_text": [hidden_triplets_text[int(idx)] for idx in hidden_triplets['triplet_ids']],
                    "resolved_entity": hidden_triplets['resolved_entity']
                }

        # get remaining triplets
        unverified_graph_triplets = self.state['resolve_entity_steps'][-1]['resolved_graph']
        verified_graph_triplets = []
        for key, value in valid_subclaims.items():
            for triplet_text in value['triplets_text']:
                verified_graph_triplet = triplet_text.replace(key, str(value['resolved_entity']))
                verified_graph_triplets.append(verified_graph_triplet)
                unverified_graph_triplets = unverified_graph_triplets.replace(verified_graph_triplet, "")
        unverified_graph_triplets = "\n".join([s for s in unverified_graph_triplets.strip().split("\n") if s])
        if len(unverified_graph_triplets) == 0: return True
        self.state['unverified_graph_triplets'] = unverified_graph_triplets
        self.state['verified_graph_triplets'] = "\n".join(verified_graph_triplets)
        
        # get subclaims
        prompt = self.prompt_loader.construct_zero_shot_subclaim_gen(claim=self.state['clarified_claim'], unverified_graph=self.state['unverified_graph_triplets'], verified_graph=self.state['verified_graph_triplets'])
        res = self.get_completion(prompt)
        self.state['remaining_subclaims'] = [{"subclaim": sc} for sc in StepParser.parse_query(res)['subclaims']]
        veracity = True
        for sc in self.state['remaining_subclaims']:
            sc_veracity = self.qa.verify_claim(sc['subclaim'], corpus_name=args.dataset, use_rerank=args.use_rerank)['veracity']
            sc.update(sc_veracity)
            if sc['veracity'] in [False, 'false', 'False', 'FALSE', 'null', None, 'None', 'NULL', 'Null']:
                veracity = False

        return veracity
        
    def check_fact(self):
        self._claim_clarify()
        self._construct_graph()
        self._resolve_entity()
        if not self._check_pass_resolve_entity():
            return False
        return self._check_subclaim()


def check_fact_job(sample, save_folder, n_results=1):
    all_results = []
    for _ in range(n_results):
        state = {
            "gold_veracity": sample['label'],
            "predicted_veracity": False,
            "global_error": None,
        }
        try:
            agent = ZeroShotFactCheckingAgent(sample["claim"])
            state['predicted_veracity'] = agent.check_fact()
        except Exception as e:
            state['global_error'] = traceback.format_exc()
        state.update(agent.state)
        all_results.append(deepcopy(state))
    sample_id = sample["id"]
    all_results = {
        "id": sample_id,
        "results": all_results
    }
    with open(os.path.join(save_folder, f"{sample_id}.json"), "w") as f:
        json.dump(all_results, f)
    return all_results

def get_current_date_time():
    return datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--llm-host-address", type=str, default="http://10.254.138.192:9002")
    parser.add_argument("--job-version", type=str, default=get_current_date_time())
    parser.add_argument("--job-folder", type=str)
    parser.add_argument("--dataset", type=str, choices=['hover', 'feverous'])
    parser.add_argument("--preprocessed-dataset-path", type=str)
    # parser.add_argument("--split", type=str, choices=['train', 'dev'])
    parser.add_argument("--partition", type=str)
    # parser.add_argument("--n-samples", type=int, default=-1)
    # parser.add_argument("--n-samples-seed", type=int, default=-1)
    
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--retrieval-topk-sparse", type=int)
    parser.add_argument("--retrieval-topk-dense", type=int)
    parser.add_argument("--retrieval-topk-rerank", type=int)
    
    parser.add_argument("--cache-completion", action="store_true")
    parser.add_argument("--use-cached-completion", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    
    parser.add_argument("--n-results", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--resolve-entity-max-loop", type=int, default=1)

    parser.add_argument("--construct-graph-examples-set-id", type=str, default=None)
    parser.add_argument("--construct-graph-examples-prompt-set-path", type=str, default=None)
    
    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # get current serving model
    args.model = requests.get(f"{args.llm_host_address}/v1/models").json()['data'][0]['id']
    pprint(vars(args))

    # prepare benchmark folder
    job_id = uuid4().hex
    save_folder = os.path.join(args.job_folder, job_id)
    os.makedirs(save_folder, exist_ok=True)
    sample_output_folder = os.path.join(save_folder, "sample_outputs")
    os.makedirs(sample_output_folder, exist_ok=True)
    
    with open(os.path.join(save_folder, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # get partition
    data_loader = DatasetLoader(args.preprocessed_dataset_path)
    partition = data_loader.get_partition(dataset_name=args.dataset, dataset_partition=args.partition)
    # if args.n_samples == -1: args.n_samples = len(partition)
    # if args.n_samples < len(partition):
    #     if args.n_samples_seed != -1:
    #         random.seed(args.n_samples_seed)
    #     random.shuffle(partition)
    # partition = partition[:args.n_samples]

    # get results
    task_dict = {sample['id']: lambda sample=sample, save_folder=save_folder, n_results=args.n_results: check_fact_job(sample, sample_output_folder, n_results) for sample in partition}
    results = multi_process_task_dict(task_dict, num_workers=args.n_workers)
    with open(os.path.join(save_folder, "results.json"), "w") as f:
        json.dump(list(results.values()), f, ensure_ascii=False)
    shutil.rmtree(sample_output_folder)
    