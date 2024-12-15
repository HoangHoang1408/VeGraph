from FlagEmbedding import FlagReranker, BGEM3FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import os
import numpy as np
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
from transformers import AutoTokenizer

# DEVICE = "cpu" # "cuda:4"
BI_ENCODER_PATH = "/workspace/home/hoangpv4/models/encoder_models/bge-m3"
CROSS_ENCODER_PATH = "/workspace/home/hoangpv4/models/encoder_models/bge-reranker-v2-m3"

CHUNK_OVERLAP = 64

BI_ENCODER_MAX_LENGTH = 1024
BI_ENCODER_TOP = 1024
BI_ENCODER_BATCH_SIZE = 256

CROSS_ENCODER_MAX_LENGTH = 512
CROSS_ENCODER_TOP = 512
CROSS_ENCODER_BATCH_SIZE = 256


bi_encoder = BGEM3FlagModel(BI_ENCODER_PATH, use_fp16=True)
bi_encoder_tokenizer = AutoTokenizer.from_pretrained(BI_ENCODER_PATH)
bi_encoder_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=BI_ENCODER_MAX_LENGTH,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=lambda x: len(bi_encoder_tokenizer(x)['input_ids']),
)
def rerank_with_bi_encoder(query, text_list, top=BI_ENCODER_TOP, max_length=BI_ENCODER_MAX_LENGTH):
    id_to_text = {i: text for i, text in enumerate(text_list)}
    text_to_id = {}
    for k, v in id_to_text.items():
        for t in bi_encoder_text_splitter.split_text(v):
            text_to_id[t] = k
    text_list = list(text_to_id.keys())
    query_embedding = bi_encoder.encode(query, batch_size=BI_ENCODER_BATCH_SIZE, max_length=BI_ENCODER_MAX_LENGTH)['dense_vecs']
    doc_embeddings = bi_encoder.encode(text_list, batch_size=BI_ENCODER_BATCH_SIZE, max_length=BI_ENCODER_MAX_LENGTH)['dense_vecs']
    scores = (query_embedding @ doc_embeddings.T).tolist()
    best_indices = np.argsort(scores)[::-1][:top]
    text_list = [text_list[i] for i in best_indices]
    text_list = list(set([id_to_text[text_to_id[t]] for t in text_list]))
    return text_list

cross_encoder = FlagReranker(CROSS_ENCODER_PATH, use_fp16=True)
cross_encoder_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_PATH)
cross_encoder_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CROSS_ENCODER_MAX_LENGTH,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=lambda x: len(cross_encoder_tokenizer(x)['input_ids']),
)
def rerank_with_cross_encoder(query, text_list, top=CROSS_ENCODER_TOP, max_length=CROSS_ENCODER_MAX_LENGTH):
    id_to_text = {i: text for i, text in enumerate(text_list)}
    text_to_id = {}
    for k, v in id_to_text.items():
        for t in cross_encoder_text_splitter.split_text(v):
            text_to_id[t] = k
    text_list = list(text_to_id.keys())
    scores = cross_encoder.compute_score([[query, t] for t in text_list], normalize=True, batch_size=CROSS_ENCODER_BATCH_SIZE)
    best_indices = np.argsort(scores)[::-1][:top]
    text_list = [text_list[i] for i in best_indices]
    text_list = list(set([id_to_text[text_to_id[t]] for t in text_list]))
    return text_list