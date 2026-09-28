from pathlib import Path
import re

from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = None

    # =========================================================
    # BUILD RAG KNOWLEDGE BASE
    # =========================================================

    def build_from_folder(self, folder_path):

        from preprocessing.document_loader import load_document
        from preprocessing.cleaner import clean_text
        from preprocessing.chunker import chunk_text

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"Reference folder not found: {folder}"
            )

        supported_extensions = {
            ".pdf",
            ".docx",
            ".pptx",
            ".txt"
        }

        files = [
            file
            for file in folder.iterdir()
            if (
                file.is_file()
                and file.suffix.lower() in supported_extensions
            )
        ]

        if not files:
            raise ValueError(
                "No supported reference files were found."
            )

        print("\n===== BUILDING RAG KNOWLEDGE BASE =====")

        all_chunks = []

        for file_path in files:

            print(
                f"\nProcessing: {file_path.name}"
            )

            try:

                raw_text = load_document(
                    str(file_path)
                )

                print(
                    f"Extracted characters: "
                    f"{len(raw_text)}"
                )

                cleaned_text = clean_text(
                    raw_text
                )

                chunks = chunk_text(
                    cleaned_text
                )

                print(
                    f"Created chunks: "
                    f"{len(chunks)}"
                )

                for chunk_id, chunk in enumerate(chunks):

                    all_chunks.append({
                        "text": chunk,
                        "source": file_path.name,
                        "chunk_id": chunk_id
                    })

            except Exception as error:

                print(
                    f"ERROR processing "
                    f"{file_path.name}:"
                )

                print(error)

        if not all_chunks:

            raise ValueError(
                "No usable text was extracted "
                "from the reference files."
            )

        print("\nCreating embeddings...")

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = (
            self.embedding_model.encode(
                texts
            )
        )

        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(
            dimension
        )

        self.vector_store.add(
            embeddings,
            all_chunks
        )

        print(
            "\n===== RAG BUILD COMPLETE ====="
        )

        print(
            "Reference files:",
            len(files)
        )

        print(
            "Total chunks:",
            len(all_chunks)
        )

        print(
            "Vector dimension:",
            dimension
        )

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, storage_path):

        if self.vector_store is None:

            raise RuntimeError(
                "Build the RAG knowledge base first."
            )

        self.vector_store.save(
            storage_path
        )

        print(
            f"\nRAG knowledge base saved to: "
            f"{storage_path}"
        )

    # =========================================================
    # LOAD
    # =========================================================

    def load(self, storage_path):

        storage = Path(
            storage_path
        )

        if not storage.exists():

            raise FileNotFoundError(
                f"RAG storage not found: {storage}"
            )

        self.vector_store = (
            VectorStore.load(
                storage
            )
        )

        print(
            f"\nRAG knowledge base loaded from: "
            f"{storage}"
        )

    # =========================================================
    # TOKENIZATION
    # =========================================================

    def _tokenize(self, text):

        return re.findall(
            r"[a-zA-Z0-9]+(?:\*)?",
            text.lower()
        )

    # =========================================================
    # IMPORTANT QUERY TOKENS
    # =========================================================

    def _important_tokens(self, query):

        tokens = self._tokenize(
            query
        )

        generic_words = {

            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",

            "of",
            "for",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "from",
            "with",
            "by",

            "using",
            "used",
            "based",

            "algorithm",
            "algorithms",

            "method",
            "methods",

            "system",
            "systems",

            "search",

            "problem",
            "problems",

            "approach",
            "approaches",

            "technique",
            "techniques",

            "process",
            "processes",

            "model",
            "models",

            "concept",
            "concepts",

            "example",
            "examples"
        }

        important = [
            token
            for token in tokens
            if (
                token not in generic_words
                and len(token) > 1
            )
        ]

        return important

    # =========================================================
    # KEYWORD SCORE
    # =========================================================

    def _keyword_score(
        self,
        query,
        document
    ):

        important_tokens = (
            self._important_tokens(
                query
            )
        )

        if not important_tokens:
            return 0.0

        text = document["text"].lower()

        matched = 0

        for token in important_tokens:

            if token in text:
                matched += 1

        if matched == 0:
            return 0.0

        return (
            matched
            / len(important_tokens)
        )

    # =========================================================
    # KEYWORD SEARCH
    # =========================================================

    def _keyword_search(
        self,
        query,
        max_candidates=100
    ):

        important_tokens = (
            self._important_tokens(
                query
            )
        )

        if not important_tokens:
            return []

        results = []

        for document in (
            self.vector_store.documents
        ):

            text_lower = (
                document["text"].lower()
            )

            matched = 0

            for token in important_tokens:

                if token in text_lower:
                    matched += 1

            if matched == 0:
                continue

            keyword_score = (
                matched
                / len(important_tokens)
            )

            results.append({

                "document": document,

                "keyword_score": (
                    keyword_score
                )
            })

        results.sort(
            key=lambda item:
                item["keyword_score"],
            reverse=True
        )

        return results[
            :max_candidates
        ]

    # =========================================================
    # CONTENT QUALITY SCORE
    # =========================================================

    def _content_quality_score(
        self,
        document
    ):

        text = (
            document["text"]
            .lower()
        )

        score = 1.0

        # -----------------------------------------------------
        # Strong penalty for index-like content
        # -----------------------------------------------------

        first_part = text[:700]

        index_indicators = [
            "index",
            "chapter ",
            "page ",
            "contents",
            "references",
            "bibliography"
        ]

        for indicator in index_indicators:

            if indicator in first_part:
                score -= 0.25

        # -----------------------------------------------------
        # Strong penalty for bibliography/reference content
        # -----------------------------------------------------

        bibliography_indicators = [
            "technical report",
            "journal",
            "proceedings",
            "conference",
            "vol.",
            "pp.",
            "university"
        ]

        bibliography_matches = 0

        for indicator in bibliography_indicators:

            if indicator in text:
                bibliography_matches += 1

        score -= min(
            bibliography_matches * 0.08,
            0.40
        )

        # -----------------------------------------------------
        # Reward explanatory content
        # -----------------------------------------------------

        explanatory_terms = [

            "is called",
            "is defined",
            "evaluates",
            "the algorithm",
            "the function",
            "the idea",
            "the cost",
            "we define",
            "can be used",
            "works by",
            "therefore",
            "because",
            "this means",
            "the basis",
            "choosing the",
            "given by"
        ]

        for term in explanatory_terms:

            if term in text:
                score += 0.08

        # -----------------------------------------------------
        # Reward mathematical/formula explanations
        # -----------------------------------------------------

        if "f(n)" in text:
            score += 0.15

        if "g(n)" in text:
            score += 0.10

        if "h(n)" in text:
            score += 0.10

        # -----------------------------------------------------
        # Keep score within valid range
        # -----------------------------------------------------

        return max(
            0.0,
            min(score, 1.0)
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query,
        k=5,
        min_score=0.40
    ):

        if self.vector_store is None:

            raise RuntimeError(
                "RAG vector store has not "
                "been built or loaded yet."
            )

        # =====================================================
        # 1. Encode query
        # =====================================================

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )[0]
        )

        # =====================================================
        # 2. Semantic search
        # =====================================================

        candidate_k = min(
            20,
            len(
                self.vector_store.documents
            )
        )

        semantic_results = (
            self.vector_store.search(
                query_embedding,
                k=candidate_k
            )
        )

        # =====================================================
        # 3. Keyword search
        # =====================================================

        keyword_results = (
            self._keyword_search(
                query,
                max_candidates=100
            )
        )

        # =====================================================
        # 4. Combine candidates
        # =====================================================

        combined = {}

        # -----------------------------------------------------
        # Semantic candidates
        # -----------------------------------------------------

        for result in semantic_results:

            document = (
                result["document"]
            )

            key = (
                document["source"],
                document["chunk_id"]
            )

            combined[key] = {

                "document": document,

                "semantic_score": (
                    result["score"]
                ),

                "keyword_score": (
                    self._keyword_score(
                        query,
                        document
                    )
                )
            }

        # -----------------------------------------------------
        # Keyword candidates
        # -----------------------------------------------------

        for result in keyword_results:

            document = (
                result["document"]
            )

            key = (
                document["source"],
                document["chunk_id"]
            )

            if key not in combined:

                combined[key] = {

                    "document": document,

                    "semantic_score": 0.0,

                    "keyword_score": (
                        result[
                            "keyword_score"
                        ]
                    )
                }

            else:

                combined[key][
                    "keyword_score"
                ] = (
                    result[
                        "keyword_score"
                    ]
                )

        # =====================================================
        # 5. Semantic reranking
        # =====================================================

        rerank_candidates = list(
            combined.values()
        )

        keyword_only = [

            item

            for item in rerank_candidates

            if item[
                "semantic_score"
            ] == 0.0
        ]

        if keyword_only:

            keyword_texts = [

                item[
                    "document"
                ]["text"]

                for item in keyword_only
            ]

            keyword_embeddings = (
                self.embedding_model.encode(
                    keyword_texts
                )
            )

            index = 0

            for item in keyword_only:

                item[
                    "semantic_score"
                ] = float(

                    keyword_embeddings[
                        index
                    ]

                    @ query_embedding
                )

                index += 1

        # =====================================================
        # 6. Calculate final score
        # =====================================================

        for item in rerank_candidates:

            semantic_score = (
                item[
                    "semantic_score"
                ]
            )

            keyword_score = (
                item[
                    "keyword_score"
                ]
            )

            content_quality = (
                self._content_quality_score(
                    item["document"]
                )
            )

            item[
                "content_quality"
            ] = content_quality

            # -------------------------------------------------
            # Final ranking
            #
            # Semantic relevance = 60%
            # Keyword relevance = 20%
            # Content quality   = 20%
            # -------------------------------------------------

            item[
                "combined_score"
            ] = (

                0.60
                * semantic_score

                +

                0.20
                * keyword_score

                +

                0.20
                * content_quality
            )

        # =====================================================
        # 7. Sort
        # =====================================================

        rerank_candidates.sort(

            key=lambda item:
                item["combined_score"],

            reverse=True
        )

        # =====================================================
        # 8. Filter
        # =====================================================

        filtered_results = []

        for item in rerank_candidates:

            semantic_score = (
                item[
                    "semantic_score"
                ]
            )

            keyword_score = (
                item[
                    "keyword_score"
                ]
            )

            if (

                semantic_score
                >= min_score

                or

                keyword_score
                > 0
            ):

                filtered_results.append(
                    item
                )

        # =====================================================
        # 9. Return Top-K
        # =====================================================

        return filtered_results[
            :k
        ]