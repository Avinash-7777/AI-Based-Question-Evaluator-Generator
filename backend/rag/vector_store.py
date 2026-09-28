import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:

    def __init__(self, dimension=None):

        self.index = None
        self.documents = []

        if dimension is not None:
            self.index = faiss.IndexFlatIP(dimension)


    def add(self, embeddings, documents):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

        self.index.add(embeddings)

        self.documents.extend(documents)


    def search(self, query_embedding, k=5):

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "document": self.documents[index],
                "score": float(score)
            })

        return results


    def save(self, folder_path):

        folder = Path(folder_path)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(folder / "index.faiss")
        )

        with open(
            folder / "documents.pkl",
            "wb"
        ) as file:

            pickle.dump(
                self.documents,
                file
            )


    @classmethod
    def load(cls, folder_path):

        folder = Path(folder_path)

        store = cls()

        store.index = faiss.read_index(
            str(folder / "index.faiss")
        )

        with open(
            folder / "documents.pkl",
            "rb"
        ) as file:

            store.documents = pickle.load(file)

        return store