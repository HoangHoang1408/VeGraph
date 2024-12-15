import requests
import random

HOSTS = [
    "http://10.254.138.189:8810",
    "http://10.254.138.189:8811",
    "http://10.254.138.189:8812",
    "http://10.254.138.189:8813",
    # "http://10.254.138.189:8814",
    # "http://10.254.138.189:8815",
    # "http://10.254.138.190:8810",
    # "http://10.254.138.190:8811",
    # "http://10.254.138.190:8812",
    # "http://10.254.138.190:8813",
    # "http://10.254.138.190:8814",
    # "http://10.254.138.190:8815",
    # "http://10.254.138.190:8816",
    # "http://10.254.138.190:8817",
]

class Retriever:
    @staticmethod
    def search(
        query, 
        top_k=10, 
        corpus_name="musique", 
        hosts=HOSTS
    ):
        res = requests.post(
            random.choice(hosts) + "/retrieve",
            json=dict(
                query_text=query,
                corpus_name=corpus_name,
                retrieval_method="retrieve_from_elasticsearch",
                max_hits_count=top_k,
                document_type="title_paragraph_text",
            ),
            headers={"Content-Type": "application/json"},
        )
        res = res.json()
        chunks = [r["paragraph_text"] for r in res["retrieval"]]
        return chunks

    @staticmethod
    def search_with_reranker(
        query, 
        sparse_max_hits_count=5000, 
        biencoder_max_hits_count=1000,
        crossencoder_max_hits_count=15,
        corpus_name="hover", 
        hosts=HOSTS
    ):
        res = requests.post(
            random.choice(hosts) + "/retrieve_and_rerank",
            json=dict(
                query_text=query,
                sparse_max_hits_count=sparse_max_hits_count,
                sparse_document_type="title_paragraph_text",
                corpus_name=corpus_name,
                biencoder_max_hits_count=biencoder_max_hits_count,
                crossencoder_max_hits_count=crossencoder_max_hits_count,
                retrieval_method="retrieve_from_elasticsearch"
            ),
            headers={"Content-Type": "application/json"},
        )
        res = res.json()
        chunks = [r["paragraph_text"] for r in res["retrieval"]]
        return chunks