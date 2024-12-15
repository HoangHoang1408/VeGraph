import os
import heapq
from hashlib import sha256
from transformers import AutoTokenizer

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
import os

### completion
from openai import OpenAI
from transformers import AutoTokenizer
import random

DEFAULT_HOST = "http://10.254.138.192:9002"
CACHE_DIR = "/workspace/home/hoangpv4/graphrag/utils/response_caches"
MODEL = "Meta-Llama-3.1-70B-Instruct"
MIN_NUM_OF_CACHES=50000
MAX_TOKENS = 2048
TOP_P = 0.9
TEMPERATURE = 0.0

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join("/workspace/home/NLP_CORE/HUB_LLM", MODEL)
)

def remove_least_recently_changed_files(cache_dir=CACHE_DIR, model=MODEL, min_files_to_keep=MIN_NUM_OF_CACHES, check_prob=0.01):
    if random.random() > check_prob:
        return
        
    model_cache_dir = os.path.join(cache_dir, model)
    files = [f for f in os.listdir(model_cache_dir) if os.path.isfile(os.path.join(model_cache_dir,f))]
    if len(files) <= min_files_to_keep:
        return

    num_files_to_remove = len(files) - min_files_to_keep
    files_with_mtime = [(f, os.path.getmtime(os.path.join(model_cache_dir, f))) for f in files]
    least_recent_files = heapq.nsmallest(num_files_to_remove, files_with_mtime, key=lambda x: x[1])

    for file, _ in least_recent_files:
        os.remove(os.path.join(model_cache_dir, file))

def cache_prompt(input_prompt, prompt_output, model_name=MODEL, cache_dir=CACHE_DIR):
    model_cache_dir = os.path.join(cache_dir, model_name)
    if not os.path.exists(model_cache_dir):
        os.makedirs(model_cache_dir, exist_ok=True)
    hash_value = sha256()
    hash_value.update(input_prompt.encode())
    hash_value = hash_value.hexdigest()
    cache_path = os.path.join(model_cache_dir, f"{hash_value}.txt")
    with open(cache_path, "w") as f:
        f.write(prompt_output)
    remove_least_recently_changed_files()

def get_cached_prompt(input_prompt, model_name=MODEL, cache_dir=CACHE_DIR):
    model_cache_dir = os.path.join(cache_dir, model_name)
    hash_value = sha256()
    hash_value.update(input_prompt.encode())
    hash_value = hash_value.hexdigest()
    cache_path = os.path.join(model_cache_dir, f"{hash_value}.txt")

    if not os.path.exists(cache_path):
        return None
    
    with open(cache_path, "r") as f:
        return f.read()

def get_completion(input_prompt, system_prompt=None, use_cached_result=False, cache_result=False, model=MODEL, tokenizer=tokenizer, stream=False, temperature=TEMPERATURE, host=DEFAULT_HOST):
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{host}/v1",
    )
    if use_cached_result:
        cache = get_cached_prompt(input_prompt)
        if cache is not None:
            return cache
            
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": input_prompt})
    prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    prompt_output = client.completions.create(
        model=model,
        prompt=prompt,
        stream=stream,
        seed=148,
        temperature=temperature,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    ).choices[0].text

    if cache_result:
        cache_prompt(input_prompt, prompt_output)
        
    return prompt_output
