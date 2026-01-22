import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search using normalized scores.
    """

    def __init__(self, bm25, vector, alpha: float = 0.6):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        self.bm25 = bm25
        self.vector = vector
        self.alpha = alpha

    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        max_score = max(scores)
        if max_score == 0:
            return scores
        return [s / max_score for s in scores]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        try:
            bm25_results = self.bm25.search(query, top_k)
            vector_results = self.vector.search(query, top_k)
        except Exception as e:
            logger.error("Hybrid search failed", exc_info=True)
            raise

        # Fallback logic
        if not bm25_results:
            logger.warning("BM25 returned no results, using vector search only")
            return vector_results

        if not vector_results:
            logger.warning("Vector search returned no results, using BM25 only")
            return bm25_results

        bm25_docs, bm25_scores = zip(*bm25_results)
        vec_docs, vec_distances = zip(*vector_results)

        bm25_norm = self._normalize(list(bm25_scores))
        vec_sim = [1 / (1 + d) for d in vec_distances]
        vec_norm = self._normalize(vec_sim)

        combined = {}

        for doc, score in zip(bm25_docs, bm25_norm):
            combined[doc] = combined.get(doc, 0) + self.alpha * score

        for doc, score in zip(vec_docs, vec_norm):
            combined[doc] = combined.get(doc, 0) + (1 - self.alpha) * score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
