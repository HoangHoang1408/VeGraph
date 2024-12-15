from retriever import Retriever
from utils.json_parser import JSONParser
from prompt_loader import PromptLoader
from functools import partial
from transformers import AutoTokenizer
import os
MODEL = "Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(os.path.join("/workspace/home/NLP_CORE/HUB_LLM", MODEL))

class QA:
    def __init__(self, get_completion):
        self.retriever = Retriever()
        self.prompt_loader = PromptLoader()
        self.get_completion = get_completion
        
    def answer_question(self, question, corpus_name, use_rerank=False, max_context_tokens=5800):
        if use_rerank:
            search_function = self.retriever.search_with_reranker
        else:
            search_function = self.retriever.search
        search_results = search_function(query=question, corpus_name=corpus_name)
        context = "\n\n".join(search_results)
        encoded = tokenizer.encode(context, add_special_tokens=False)
        if len(encoded) > max_context_tokens:
            context = tokenizer.decode(encoded[:max_context_tokens])
        prompt = self.prompt_loader.construct_qa_prompt(question=question, context=context)
        completion = self.get_completion(prompt)
        try:
            answer = JSONParser.extract_json_dict(completion)['answer']
        except Exception as e:
            print("Before:", completion)
            answer = completion.split(":")[1].strip("}").strip()
            print("Fixed to:", answer)
        return {
            "answer": answer,
            "search_results": search_results
        }
        
    def verify_claim(self, claim, corpus_name, use_rerank=False, max_context_tokens=5800):
        if use_rerank:
            search_function = self.retriever.search_with_reranker
        else:
            search_function = self.retriever.search
        search_results = search_function(query=claim, corpus_name=corpus_name)
        context = "\n\n".join(search_results)
        encoded = tokenizer.encode(context, add_special_tokens=False)
        if len(encoded) > max_context_tokens:
            context = tokenizer.decode(encoded[:max_context_tokens])
        prompt = self.prompt_loader.construct_fact_check_prompt(claim=claim, context=context)
        completion = self.get_completion(prompt)
        try:
            veracity = JSONParser.extract_json_dict(completion)
        except Exception as e:
            print("Before:", completion)
            veracity = False
            if "true" in " ".join(completion.split("veracity")[1:]):
                veracity = True
            try:
                rationale = completion.split("veracity")[0].split("rationale")[1]
            except:
                rationale = None
            veracity = {
                "rationale": rationale,
                "veracity": veracity
            }
            print("Fixed to:", veracity)
        return {
            "veracity": veracity,
            "search_results": search_results
        }
        