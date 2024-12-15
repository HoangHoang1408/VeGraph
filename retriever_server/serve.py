import os
from time import perf_counter
from fastapi import FastAPI, Request
from unified_retriever import UnifiedRetriever
from rerank import rerank_with_bi_encoder, rerank_with_cross_encoder

retriever = UnifiedRetriever(host="http://10.254.138.189", port=9200)
app = FastAPI()

@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}

@app.post("/retrieve")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    start_time = perf_counter()
    retrieval = getattr(retriever, retrieval_method)(**arguments)
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}

@app.post("/retrieve_and_rerank")
async def retrieve_and_rerank(arguments: Request):
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    start_time = perf_counter()
    docs = getattr(retriever, retrieval_method)(
        **dict(
            query_text=arguments["query_text"],
            max_hits_count=arguments["sparse_max_hits_count"],
            document_type=arguments["sparse_document_type"],
            corpus_name=arguments["corpus_name"],
        )
    )
    docs_text = [doc["title"] + "\n" + doc["paragraph_text"] for doc in docs]
    docs_text = rerank_with_bi_encoder(arguments["query_text"], docs_text, top=arguments['biencoder_max_hits_count'])
    docs_text = rerank_with_cross_encoder(arguments["query_text"], docs_text, top=arguments['crossencoder_max_hits_count'])
    reranked_docs = [{"paragraph_text": text} for text in docs_text]
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    return {
        "retrieval": reranked_docs,
        "time_in_seconds": time_in_seconds,
    }