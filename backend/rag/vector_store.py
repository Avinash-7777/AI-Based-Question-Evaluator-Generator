import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(self, query_embedding, k=3):
        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "document": self.documents[index],
                "distance": float(distance)
            })

        return results