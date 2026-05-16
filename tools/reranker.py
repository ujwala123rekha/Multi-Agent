from typing import List, Dict
from sentence_transformers import CrossEncoder

class RerankerTool:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # BAAI/bge-reranker-base is also great, but MiniLM is faster for local dev
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
        """Rerank documents based on semantic relevance to the query."""
        if not documents:
            return []
            
        pairs = [[query, doc['content']] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['score'] = float(score)
            
        # Sort and return top_k
        reranked = sorted(documents, key=lambda x: x['score'], reverse=True)
        return reranked[:top_k]
