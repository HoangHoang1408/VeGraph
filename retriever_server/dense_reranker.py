from sentence_transformers import SentenceTransformer, util

EMBEDDING_RERANKER = "dense_model"


class DenseReranker:
    def __init__(self, reranker_model=EMBEDDING_RERANKER):
        self.model = SentenceTransformer(reranker_model)

    def rerank(self, query, docs, top_k=10):
        query_embedding = self.model.encode([query],convert_to_tensor=True).to('cuda')
        query_embedding = util.normalize_embeddings(query_embedding)
        doc_embeddings = self.model.encode(docs,convert_to_tensor=True).to('cuda')
        doc_embeddings = util.normalize_embeddings(doc_embeddings)
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
        scores = scores.cpu().numpy()[0]
        sorted_indices = scores.argsort()[::-1][:top_k]
        return [
            {"doc_id": doc_id, "dense_score": float(scores[doc_id])}
            for doc_id in sorted_indices
        ]
