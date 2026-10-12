
import os
import re
import pickle
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        print(f"[RAG] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.documents = []

    # ============================================================
    # LOAD VECTOR STORE
    # ============================================================

    def load(self, storage_path: str = "rag_storage"):
        index_path = os.path.join(storage_path, "index.faiss")
        documents_path = os.path.join(storage_path, "documents.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not os.path.exists(documents_path):
            raise FileNotFoundError(
                f"Documents file not found: {documents_path}"
            )

        self.index = faiss.read_index(index_path)

        with open(documents_path, "rb") as file:
            self.documents = pickle.load(file)

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS/document count mismatch: "
                f"{self.index.ntotal} vectors and "
                f"{len(self.documents)} documents."
            )

        print(f"[RAG] Knowledge base loaded from: {storage_path}")
        print(f"[RAG] Total chunks: {len(self.documents)}")
        print(f"[RAG] FAISS vectors: {self.index.ntotal}")

    # ============================================================
    # TEXT HELPERS
    # ============================================================

    @staticmethod
    def _get_text(document: Any) -> str:
        if isinstance(document, dict):
            return str(document.get("text", "")).strip()

        return str(getattr(document, "text", "")).strip()

    @staticmethod
    def _important_tokens(text: str) -> List[str]:
        # Correctly supports normal words, hyphenated terms and apostrophes.
        tokens = re.findall(
            r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
            text.lower(),
        )

        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in",
            "on", "for", "with", "from", "by", "is", "are",
            "was", "were", "be", "been", "being", "this",
            "that", "these", "those", "what", "which", "who",
            "how", "why", "when", "where", "does", "do", "did",
            "can", "could", "would", "should", "will", "may",
            "might", "into", "as", "at", "it", "its", "their",
            "there", "than", "then", "describe", "explain",
            "discuss", "state", "identify", "following",
            "question", "questions", "topic", "subject",
        }

        return [
            token for token in tokens
            if token not in stopwords and len(token) > 1
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    # ============================================================
    # KEYWORD SCORE
    # ============================================================

    def _keyword_score(self, query: str, text: str) -> float:
        query_tokens = set(self._important_tokens(query))

        if not query_tokens:
            return 0.0

        text_tokens = set(self._important_tokens(text))

        matches = query_tokens.intersection(text_tokens)

        return len(matches) / len(query_tokens)

    # ============================================================
    # PHRASE SCORE
    # ============================================================

    def _phrase_score(self, query: str, text: str) -> float:
        query_lower = self._normalize_text(query)
        text_lower = self._normalize_text(text)

        # Score phrases that actually occur in the query.
        query_tokens = self._important_tokens(query)

        phrases = set(query_tokens)

        # Include adjacent query-token pairs to capture useful phrases.
        for index in range(len(query_tokens) - 1):
            phrases.add(
                query_tokens[index] + " " + query_tokens[index + 1]
            )

        if not phrases:
            return 0.0

        matches = sum(
            1 for phrase in phrases
            if phrase in text_lower
        )

        return matches / len(phrases)

    # ============================================================
    # CONTENT QUALITY SCORE
    # ============================================================

    def _content_quality_score(self, text: str) -> float:
        text_lower = text.lower()
        score = 0.65

        # Penalize obvious non-learning material, but do not discard
        # a technical passage merely because it contains one such term.
        negative_patterns = [
            "table of contents",
            "bibliography",
            "bibliographical notes",
            "historical notes",
            "subject index",
            "author index",
            "index of",
        ]

        for pattern in negative_patterns:
            if pattern in text_lower:
                score -= 0.20

        if len(text.split()) < 20:
            score -= 0.15

        if len(text.split()) > 40:
            score += 0.10

        technical_patterns = [
            "algorithm",
            "model",
            "method",
            "example",
            "equation",
            "probability",
            "classification",
            "prediction",
            "learning",
            "training",
            "inference",
            "decision",
            "search",
            "knowledge",
            "reasoning",
            "neural",
            "network",
            "optimization",
            "data",
        ]

        technical_hits = sum(
            1 for pattern in technical_patterns
            if pattern in text_lower
        )

        score += min(technical_hits * 0.025, 0.25)

        return max(0.0, min(score, 1.0))

    # ============================================================
    # SEMANTIC SCORE NORMALIZATION
    # ============================================================

    def _normalize_semantic_score(self, raw_score: float) -> float:
        """
        Convert FAISS results to a similarity-like score in [0, 1].

        IndexFlatIP with normalized vectors produces inner-product
        similarity. For L2 indexes, smaller distances are better.
        """

        metric_type = getattr(self.index, "metric_type", None)

        if metric_type == faiss.METRIC_L2:
            distance = max(0.0, float(raw_score))

            # Normalized vectors have squared L2 distance in [0, 4].
            return max(0.0, min(1.0 - distance / 2.0, 1.0))

        # Inner-product similarity may be negative.
        return max(0.0, min(float(raw_score), 1.0))

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.20,
    ) -> List[Dict[str, Any]]:

        if self.index is None:
            raise RuntimeError(
                "RAG index is not loaded. Call retriever.load() first."
            )

        if not self.documents:
            return []

        query = str(query).strip()

        if not query or k <= 0:
            return []

        # --------------------------------------------------------
        # ENCODE QUERY
        # --------------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # SEARCH A LARGER CANDIDATE POOL
        # --------------------------------------------------------

        search_k = min(
            len(self.documents),
            max(k * 30, 200),
        )

        raw_scores, indices = self.index.search(
            query_embedding,
            search_k,
        )

        raw_scores = raw_scores[0]
        indices = indices[0]

        # --------------------------------------------------------
        # SCORE CANDIDATES
        # --------------------------------------------------------

        results = []

        for raw_score, index_position in zip(raw_scores, indices):
            if index_position < 0:
                continue

            if index_position >= len(self.documents):
                continue

            document = self.documents[index_position]
            text = self._get_text(document)

            if not text:
                continue

            semantic_score = self._normalize_semantic_score(
                float(raw_score)
            )

            keyword_score = self._keyword_score(query, text)
            phrase_score = self._phrase_score(query, text)
            content_quality = self._content_quality_score(text)

            # General-purpose ranking: no ID3-only boosts.
            combined_score = (
                0.65 * semantic_score
                + 0.20 * keyword_score
                + 0.05 * phrase_score
                + 0.10 * content_quality
            )

            combined_score = max(
                0.0,
                min(combined_score, 1.0),
            )

            if combined_score < min_score:
                continue

            results.append({
                "document": document,
                "semantic_score": round(semantic_score, 4),
                "keyword_score": round(keyword_score, 4),
                "phrase_score": round(phrase_score, 4),
                "content_quality": round(content_quality, 4),
                "combined_score": round(combined_score, 4),
            })

        results.sort(
            key=lambda item: item["combined_score"],
            reverse=True,
        )

        # --------------------------------------------------------
        # REMOVE DUPLICATE CHUNKS
        # --------------------------------------------------------

        final_results = []
        seen_texts = set()

        for result in results:
            text = self._get_text(result["document"])
            normalized_text = self._normalize_text(text)

            if not normalized_text or normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)
            final_results.append(result)

            if len(final_results) >= k:
                break

        print(
            f"[RAG] Query={query!r} | "
            f"candidates={len(results)} | "
            f"returned={len(final_results)} | "
            f"requested={k}"
        )

        for result in final_results[:5]:
            document = result["document"]

            if isinstance(document, dict):
                source = document.get("source", "unknown")
                chunk_id = document.get("chunk_id", "unknown")
            else:
                source = getattr(document, "source", "unknown")
                chunk_id = getattr(document, "chunk_id", "unknown")

            print(
                f"[RAG] score={result['combined_score']:.3f} "
                f"source={source} chunk={chunk_id}"
            )

        return final_results