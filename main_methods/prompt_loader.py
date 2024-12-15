import os
import json
from prompts.qa import QA_PROMPT_WITH_DOCS
from prompts.fact_check import FACT_CHECK_WITH_DOCS
from prompts.few_shot_claim_clarify import FEW_SHOT_CLAIM_CLARIFY
from prompts.few_shot_construct_graph import FEW_SHOT_CONSTRUCT_GRAPH, FEW_SHOT_CONSTRUCT_GRAPH_EXAMPLE_TEMPLATE
from prompts.few_shot_query_gen import FEW_SHOT_QUERY_GEN
from prompts.zero_shot_claim_clarify import ZERO_SHOT_CLAIM_CLARIFY
from prompts.zero_shot_construct_graph import ZERO_SHOT_CONSTRUCT_GRAPH
from prompts.zero_shot_query_gen import ZERO_SHOT_QUERY_GEN
from prompts.zero_shot_query_refine import ZERO_SHOT_QUERY_REFINE, ZERO_SHOT_QUERY_REFINE_FAILED_TRIAL_TEMPLATE
from prompts.zero_shot_subclaim_gen import ZERO_SHOT_SUBCLAIM_GEN
from utils.load_data import LoadData
from pathlib import Path

class PromptLoader:
    def __init__(self, log_prompt=False):
        self.log_prompt = log_prompt
        self.examples = {"hover": {}, "feverous": {}}
        examples_dir = Path(__file__).parent / "prompts" / "examples"
        for name in ['hover', 'feverous']:
            self.examples[name]['claim_clarify'] = LoadData.load_data(str(examples_dir / "claim_clarify" / name / "examples.json"))
            self.examples[name]['construct_graph'] = LoadData.load_data(str(examples_dir / "construct_graph" / name / "examples.json"))
    
    def return_prompt(self, prompt):
        if self.log_prompt:
            print(prompt)
        return prompt

    def construct_qa_prompt(self, question, context):
        prompt = QA_PROMPT_WITH_DOCS.replace("{{question}}", question).replace("{{context}}", context)
        return self.return_prompt(prompt)

    def construct_fact_check_prompt(self, claim, context):
        prompt = FACT_CHECK_WITH_DOCS.replace("{{claim}}", claim).replace("{{context}}", context)
        return self.return_prompt(prompt)

    def construct_few_shot_construct_graph_prompt(self, claim, dataset_name, prompt_set_path, example_set_id):
        folder_path = prompt_set_path
        with open(os.path.join(folder_path, f"{example_set_id}.json")) as f:
            examples = json.load(f)
        example_text = []
        for e in examples:
            example_text.append(
                FEW_SHOT_CONSTRUCT_GRAPH_EXAMPLE_TEMPLATE.replace("{{claim}}", e['claim']).replace("{{graph}}", e['graph']).replace("{{guidance_for_graph_construction}}", e['guidance_for_graph_construction'])
            )
        example_text = "\n\n".join(example_text)
        prompt = FEW_SHOT_CONSTRUCT_GRAPH.replace("{{examples_text}}", example_text).replace("{{claim}}", claim)
        return self.return_prompt(prompt)

    def construct_zero_shot_construct_graph_prompt(self, claim):
        prompt = ZERO_SHOT_CONSTRUCT_GRAPH.replace("{{claim}}", claim)
        return self.return_prompt(prompt)

    def construct_zero_shot_query_gen_prompt(self, claim, graph):
        prompt = ZERO_SHOT_QUERY_GEN.replace("{{claim}}", claim).replace("{{graph}}", graph)
        return self.return_prompt(prompt)

    def construct_zero_shot_subclaim_gen(self, claim, verified_graph, unverified_graph):
        prompt = ZERO_SHOT_SUBCLAIM_GEN.replace("{{claim}}", claim).replace("{{verified_graph}}", verified_graph).replace("{{unverified_graph}}", unverified_graph)
        return self.return_prompt(prompt)

    def construct_zero_shot_query_refine_prompt(self, claim, graph, failed_trails):
        failed_trails_text = []
        for i, trail in enumerate(failed_trails):
            failed_trails_text.append(
                ZERO_SHOT_QUERY_REFINE_FAILED_TRIAL_TEMPLATE.replace("{{i}}", str(i)).replace("{{rationale}}", trail['rationale']).replace("{{question}}", trail['question'])
            )
        failed_trails_text = "\n\n".join(failed_trails_text)
        prompt = ZERO_SHOT_QUERY_REFINE.replace("{{failed_trials_text}}", failed_trails_text).replace("{{claim}}", claim).replace("{{graph}}", graph)
        return self.return_prompt(prompt)

