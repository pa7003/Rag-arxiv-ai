class HybridRetriever:
    def __init__(self, bm25, vector, alpha=0.6):
        self.bm25 = bm25
        self.vector = vector
        self.alpha = alpha

    def search(self, query, top_k=5):
        bm25_results = self.bm25.search(query, top_k)
        vector_results = self.vector.search(query, top_k)

        scores = {}

        for doc, s in bm25_results:
            scores[doc] = scores.get(doc, 0) + self.alpha * s

        for doc, d in vector_results:
            sim = 1 / (1 + d)
            scores[doc] = scores.get(doc, 0) + (1 - self.alpha) * sim

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
