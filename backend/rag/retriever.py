from .document_loader import load_text_files
from .chunker import chunk_text
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = None

    def build(self, knowledge_base_path):

        # 1. Load documents
        documents = load_text_files(
            knowledge_base_path
        )

        # 2. Split documents into chunks
        chunks = []

        for document in documents:

            document_chunks = chunk_text(
                document["text"]
            )

            for chunk in document_chunks:

                chunks.append({
                    "text": chunk,
                    "source": document["source"]
                })

        # 3. Convert chunks into embeddings
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts
        )

        # 4. Create vector store
        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(
            dimension
        )

        # 5. Store embeddings + chunks
        self.vector_store.add(
            embeddings,
            chunks
        )

    def search(self, query, k=3):

        # Convert query into an embedding
        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        # Search FAISS
        return self.vector_store.search(
            query_embedding,
            k
        )