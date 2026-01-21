import faiss
from sentence_transformers import SentenceTransformer

class VectorRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(documents, show_progress_bar=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        return [(self.documents[i], distances[0][pos])
                for pos, i in enumerate(indices[0])]
