import os
from argparse import ArgumentParser
from functools import partial
from pprint import pprint
from uuid import uuid4
from datetime import datetime
import json
import requests
import random
from dataset_loader import DatasetLoader
from utils.get_completion import get_completion
from utils.json_parser import JSONParser
from utils.multi_process_task_dict import multi_process_task_dict
from retriever import Retriever
from transformers import AutoTokenizer
from functools import partial

MODEL_PATH = "/workspace/home/NLP_CORE/HUB_LLM/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

QA_PROMPT = """
### Task: Based only on the information provided in the given documents, answer the following question.

### Guidelines:
1) Your response must exclusively use information from the provided documents. Do **not** rely on outside knowledge or generate knowledge yourself.
2) Return a condensed answer in the following JSON format:
{"answer": "your condensed answer for the question"}
3) If the entity is not found in the documents, return:
{"answer": "No answer found"}

### Documents:
{{context}}

### Question:
{{question}}

### Now answer the question and return in correct format.
""".strip()

class QA:
    def __init__(self):
        self.retriever = Retriever()
        
    def answer_question(self, question, corpus_name, use_rerank=True, max_context_tokens=5800):
        if use_rerank:
            search_function = self.retriever.search_with_reranker
        else:
            search_function = self.retriever.search
        search_results = search_function(query=question, corpus_name=corpus_name)
        
        context = "\n\n".join(search_results)
        encoded = tokenizer.encode(context, add_special_tokens=False)
        if len(encoded) > max_context_tokens:
            context = tokenizer.decode(encoded[:max_context_tokens])
        prompt = QA_PROMPT.replace("{{context}}", context).replace("{{question}}", question)
        completion = get_completion(prompt)
        try:
            answer = JSONParser.extract_json_dict(completion)['answer']
        except Exception as e:
            try:
                print("Before:", completion)
                answer = completion.split(":")[1].strip("}").strip()
                print("Fixed to:", answer)
            except:
                answer = completion
        return answer


QUESTION_GEN_PROMPT = """### Task: Generate necessary questions that need to be answered (in the following format) in order to verify the following claim:

-- Examples with correct returning format -- 
Claim: Howard University Hospital and Providence Hospital are both located in Washington, D.C.
>>>>>>
Followup Question: Where is Howard Hospital located?
Followup Question: Where is Providence Hospital located? 
------
Claim: An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season.
>>>>>>
Followup Question: Which Formula 1 car was designed by Peter McCool during the 2007 Formula One season?
Followup Question: Did an IndyCar driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season?
------
Claim: Sumo wrestler Toyozakura Toshiaki committed match-fixing, ending his career in 2011 that started in 1989.
>>>>>>
Followup Question: When did Sumo wrestler Toyozakura Toshiaki ended his career?
Followup Question: What is Toyozakura Toshiaki's occupation?
Followup Question: Did Sumo wrestler Toyozakura Toshiaki committed match-fixing?
------
Claim: In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago, United States, from August 27 to September 7) in Chicago, United States, and the world amateur welterweight title in Mexico City.
>>>>>>
Followup Question: When was Alfredo Cornejo Cuevas born?
Followup Question: Did Alfredo Cornejo Cuevas win the gold metal in the welterweight division at the Pan American Games in 1959?
Followup Question: Where was The Pan American Games in 1959 held?
Followup Question: Did Alfredo Cornejo Cuevas win the world amateur welterweight title in Mexico City?
------

-- Actual input claim --
Claim: {{claim}}
>>>>>>
"""

AGGREGATE_PROMPT = """### Task: Given a question and a context, provide a SUPPORTED or NOT_SUPPORTED and explain why.

-- Examples with correct returning format --
Question: 
Is it true that The writer of the song Girl Talk and Park So-yeon have both been members of a girl group. ?

Context:
Who is the writer of the song Girl Talk? Tionne Watkins is the writer of the song Girl Talk.
Is Park So-yeon a member of a girl group? Park Soyeon is a South Korean singer. She is a former member of the kids girl group I& Girls.
Is the writer of the song Girl Talk a member of a girl group? Watkins rose to fame in the early 1990s as a member of the girl-group TLC
>>>>>>
Here are the reasons, Tionne Watkins is the writer of the song Girl Talk, and she fame in the early 1990s as a member of the girl-group TLC.
Park Soyeon is a South Korean singer. She is a former member of the kids girl group I& Girls.
The claim is [SUPPORTED].
------
Question:
Is it true that A hockey team calls the 70,000 capacity Madison Square Garden it's home. That team, along with the New York Islanders, and the New Jersey Devils NHL franchise, are popular in the New York metropolitan area. ?

Context:
Which hocky team calls Madison Square Garden Home? Madison Square Garden hosts approximately 320 events a year. It is the home to the New York Rangers of the National Hockey League
What is the capacity of Madison Square Garden? Madison Square Garden has a capacity of 19.500.
Is New York Islanders popular in New York Metropolitan area? The New York Islanders are a professional ice hockey team based in Elmont, New York. ... 
>>>>>>
Here are the reasons, Madison Square Garden hosts approximately 320 events a year. It is the home to the New York Rangers of the National Hockey League and the New York Islanders are a professional ice hockey team based in Elmont, New York. Madison Square Garden has a capacity of 19.500, not 70,000.
The claim is [NOT_SUPPORTED].
------
Question: 
Is it true that Werner Gunter Jaff\u00e9 Fellner was born in Frankfurt in the German state of Hesse and the fifth-largest city in Germany. ?

Context:
Where was Werner Gunter Jaff\u00e9 Fellner born? Werner Gunter Jaff\u00e9 Fellner was born in Frankfurt.
Which state is Frankfurt in? Frankfurt is in the German state of Hesse.
>>>>>>
Here are the reasons, Werner Gunter JafFf\u00e9 Fellner was born in Frankfurt and Frankfurt is in the German state of Hesse.
The claim is [SUPPORTED].
------
Question:
Is it true that The American lyricist Tom Jones,  born in 1928, co-authored the screenplay for the musical film The Fantastics. ?

Context:
When was Tom Jones born? Thomas Jones Woodward was born in Pontypridd, South Wales, Great Britain on June 7, 1940
What is Tome Jones nationality? Sir Thomas Jones Woodward OBE is a Welsh singer. 
Who co-author the musical film The Fantastics? Tome Jones co-authored the musical film The Fantastics.
>>>>>>
Here are the reasons, Sir Thomas Jones Woodward OBE is a Welsh singer and Tome Jones co-authored the musical film The Fantastics, but Thomas Jones Woodward was born in Pontypridd, South Wales, Great Britain on June 7, 1940. Thomas Jones is British, not American.
The claim is [NOT_SUPPORTED].
------

-- Actual input question --
Question: Is it true that {{claim}}?

Context: 
{{context}}
>>>>>>
"""

QA_PROMPT = """
### Task: Based only on the information provided in the given documents, answer the following question.

### Guidelines:
1) Your response must exclusively use information from the provided documents. Do **not** rely on outside knowledge or generate knowledge yourself.
2) Return a condensed answer in the following JSON format:
{"answer": "your condensed answer for the question"}
3) If the entity is not found in the documents, return:
{"answer": "No answer found"}

### Documents:
{{context}}

### Question:
{{question}}

### Now answer the question and return in correct format.
""".strip()

def simple_approach(sample, dataset, get_completion, n_results=1, qa = QA()):
    def get_questions(claim):
        prompt = QUESTION_GEN_PROMPT.replace("{{claim}}", claim)
        res = get_completion(prompt)
        lines = res.split("\n")
        questions = [l.split("Followup Question:")[1].strip() for l in lines if "Followup Question:".lower() in l.lower()]
        return [{"question": q} for q in questions]

    def grounding(questions):
        for question in questions:
            question['answer'] = qa.answer_question(question['question'], dataset)
        return questions

    def verify(claim, answers):
        qa_context = []
        for x in answers:
            qa_context.append(f"{x['question']} {x['answer']}")
        prompt = AGGREGATE_PROMPT.replace("{{context}}", "\n".join(qa_context)).replace("{{claim}}", claim)
        completion = get_completion(prompt)
        veracity = None
        if "[SUPPORTED]" in completion:
            veracity = True
        elif "[NOT_SUPPORTED]" in completion:
            veracity = False
        if veracity is None:
            veracity = random.choice([True, False])
        return {"veracity": veracity, "rationale": completion}
    results = []
    for _ in range(n_results): 
        result = {"questions_and_answers": None, "rationale": None}
        try:
            questions = get_questions(sample['claim'])
            answers = grounding(questions)
            result.update(verify(sample['claim'], answers))
            result['questions_and_answers'] = answers
        except Exception as e:
            print(e)
            result['veracity'] = random.choice([True, False])
        results.append(result)
    with open("./temp.json", "w") as f:
        json.dump(results, f)
    sample['results'] = results
    return sample

def get_current_date_time():
    return datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--llm-host-address", type=str, default="http://10.254.138.191:9002")
    parser.add_argument("--job-version", type=str, default=get_current_date_time())
    parser.add_argument("--job-folder", type=str)
    parser.add_argument("--preprocessed-dataset-path", type=str)
    parser.add_argument("--dataset", type=str, choices=['hover', 'feverous'])
    parser.add_argument("--partition", type=str)
    
    parser.add_argument("--cache-completion", action="store_true")
    parser.add_argument("--use-cached-completion", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-results", type=int, default=1)
    
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args() 
    return args

if __name__ == "__main__":
    args = parse_args()
    get_completion = partial(get_completion, temperature=args.temperature, use_cached_result=args.use_cached_completion, cache_result=args.cache_completion, host=args.llm_host_address)
    data_loader = DatasetLoader(args.preprocessed_dataset_path)
    partition = data_loader.get_partition(args.dataset, args.partition)
    task_dict = {i: lambda sample=sample, dataset=args.dataset, get_completion=get_completion, n_results=args.n_results: simple_approach(sample, dataset, get_completion, n_results) for i, sample in enumerate(partition)}

    results = multi_process_task_dict(task_dict, num_workers=args.n_workers)
    
    job_id = uuid4().hex
    save_folder = os.path.join(args.job_folder, job_id)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "arguments.json"), "w") as f:
        json.dump(vars(args), f)
    with open(os.path.join(save_folder, "results.json"), "w") as f:
        json.dump(list(results.values()), f, ensure_ascii=False)

