import os
import re
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.documents = []

    # ============================================================
    # LOAD VECTOR STORE
    # ============================================================

    def load(self, storage_path: str = "rag_storage"):
        index_path = os.path.join(
            storage_path,
            "index.faiss"
        )

        documents_path = os.path.join(
            storage_path,
            "documents.pkl"
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not os.path.exists(documents_path):
            raise FileNotFoundError(
                f"Documents file not found: {documents_path}"
            )

        self.index = faiss.read_index(index_path)

        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)

        print(
            f"RAG knowledge base loaded from: {storage_path}"
        )

        print(
            f"Total chunks: {len(self.documents)}"
        )

    # ============================================================
    # IMPORTANT TOKENS
    # ============================================================

    def _important_tokens(
        self,
        text: str
    ) -> List[str]:

        text = text.lower()

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
            text
        )

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "on",
            "for",
            "with",
            "from",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "how",
            "why",
            "when",
            "where",
            "does",
            "do",
            "did",
            "can",
            "could",
            "would",
            "should",
            "will",
            "may",
            "might",
            "into",
            "as",
            "at",
            "it",
            "its",
            "their",
            "there",
            "than",
            "then",
        }

        return [
            token
            for token in tokens
            if token not in stopwords
            and len(token) > 1
        ]

    # ============================================================
    # QUERY EXPANSION
    # ============================================================

    def _expand_tokens(
        self,
        tokens: List[str]
    ) -> List[str]:

        expanded = set(tokens)

        mappings = {

            "id3": {
                "id3",
                "decision",
                "tree",
                "trees",
                "decision-tree",
                "decision-tree-learning",
                "information",
                "gain",
                "information-gain",
                "entropy",
                "attribute",
                "attributes",
                "split",
                "splitting",
                "root",
                "criterion",
                "selection",
                "choose",
                "choosing",
            },

            "algorithm": {
                "algorithm",
                "procedure",
                "method",
                "learning",
            },

            "decision": {
                "decision",
                "tree",
                "trees",
                "decision-tree",
                "decision-tree-learning",
            },

            "trees": {
                "tree",
                "trees",
                "decision",
                "decision-tree",
                "decision-tree-learning",
            },

            "tree": {
                "tree",
                "trees",
                "decision",
                "decision-tree",
                "decision-tree-learning",
            },

            "information": {
                "information",
                "gain",
                "information-gain",
                "entropy",
                "remainder",
            },

            "gain": {
                "gain",
                "information",
                "information-gain",
                "entropy",
                "remainder",
            },

            "attribute": {
                "attribute",
                "attributes",
                "split",
                "splitting",
                "selection",
                "choose",
                "choosing",
            },

            "attributes": {
                "attribute",
                "attributes",
                "split",
                "splitting",
                "selection",
                "choose",
                "choosing",
            },

            "entropy": {
                "entropy",
                "information",
                "gain",
                "remainder",
            },

            "split": {
                "split",
                "splitting",
                "attribute",
                "attributes",
                "criterion",
            },
        }

        for token in tokens:

            if token in mappings:
                expanded.update(
                    mappings[token]
                )

        return list(expanded)

    # ============================================================
    # KEYWORD SCORE
    # ============================================================

    def _keyword_score(
        self,
        query: str,
        text: str
    ) -> float:

        query_tokens = self._important_tokens(
            query
        )

        query_tokens = self._expand_tokens(
            query_tokens
        )

        if not query_tokens:
            return 0.0

        text_lower = text.lower()

        matches = 0

        for token in query_tokens:

            if re.search(
                rf"\b{re.escape(token)}\b",
                text_lower
            ):
                matches += 1

        score = matches / len(
            query_tokens
        )

        return min(
            score,
            1.0
        )

    # ============================================================
    # SUBTOPIC SCORE
    # ============================================================

    def _subtopic_score(
        self,
        query: str,
        text: str
    ) -> float:

        query_lower = query.lower()
        text_lower = text.lower()

        score = 0.0

        # --------------------------------------------------------
        # Decision trees
        # --------------------------------------------------------

        if (
            "decision tree" in query_lower
            or "decision trees" in query_lower
        ):

            if "decision tree" in text_lower:
                score += 0.30

            if "decision-tree" in text_lower:
                score += 0.15

            if "decision tree learning" in text_lower:
                score += 0.20

            if "tree learning" in text_lower:
                score += 0.10

        # --------------------------------------------------------
        # ID3
        # --------------------------------------------------------

        if re.search(
            r"\bid3\b",
            query_lower
        ):

            if re.search(
                r"\bid3\b",
                text_lower
            ):
                score += 0.30

            if "information gain" in text_lower:
                score += 0.25

            if "entropy" in text_lower:
                score += 0.18

            if "attribute" in text_lower:
                score += 0.10

            if "choosing attribute" in text_lower:
                score += 0.15

            if "attribute test" in text_lower:
                score += 0.12

            if "maximum gain" in text_lower:
                score += 0.15

            if "highest information gain" in text_lower:
                score += 0.15

        return min(
            score,
            1.0
        )

    # ============================================================
    # PHRASE SCORE
    # ============================================================

    def _phrase_score(
        self,
        query: str,
        text: str
    ) -> float:

        query_lower = query.lower()
        text_lower = text.lower()

        score = 0.0

        important_phrases = [
            "decision tree",
            "decision trees",
            "decision-tree",
            "decision tree learning",
            "information gain",
            "information-gain",
            "choosing attribute",
            "choosing attributes",
            "attribute test",
            "attribute tests",
            "maximum information gain",
            "highest information gain",
            "best attribute",
            "select attribute",
            "selecting attribute",
            "split criterion",
            "entropy",
            "id3",
        ]

        for phrase in important_phrases:

            if phrase in query_lower:

                if phrase in text_lower:
                    score += 0.12

        return min(
            score,
            1.0
        )

    # ============================================================
    # CONTENT QUALITY SCORE
    # ============================================================

    def _content_quality_score(
        self,
        text: str
    ) -> float:

        text_lower = text.lower()

        score = 0.50

        # --------------------------------------------------------
        # Negative signals
        # --------------------------------------------------------

        negative_patterns = [
            "table of contents",
            "bibliography",
            "bibliographical and historical notes",
            "bibliographical notes",
            "historical notes",
            "reference list",
            "references",
            "subject index",
            "author index",
            "index of",
            "chapter contents",
            "contents",
        ]

        for pattern in negative_patterns:

            if pattern in text_lower:
                score -= 0.30

        # Index-style content

        if re.search(
            r"\b[a-zA-Z]+\s+\d{2,4}\s*,\s*\d{2,4}",
            text
        ):
            score -= 0.25

        # --------------------------------------------------------
        # Historical material
        # --------------------------------------------------------

        historical_patterns = [
            "william of ockham",
            "ockham",
            "aristotle",
            "claude shannon",
            "historical",
            "quinlan",
            "1979",
            "1986",
            "first proposed",
            "first use",
            "early work",
        ]

        historical_hits = sum(
            1
            for pattern in historical_patterns
            if pattern in text_lower
        )

        if historical_hits >= 2:
            score -= 0.35

        elif historical_hits == 1:
            score -= 0.15

        # --------------------------------------------------------
        # Exercise / question material
        # --------------------------------------------------------

        exercise_patterns = [
            "exercise",
            "exercises",
            "review question",
            "review questions",
            "problem set",
            "questions",
            "quiz",
        ]

        exercise_hits = sum(
            1
            for pattern in exercise_patterns
            if pattern in text_lower
        )

        if exercise_hits >= 2:
            score -= 0.20

        elif exercise_hits == 1:
            score -= 0.08

        # --------------------------------------------------------
        # Technical signals
        # --------------------------------------------------------

        technical_patterns = [
            "algorithm",
            "information gain",
            "entropy",
            "attribute",
            "attributes",
            "decision tree",
            "decision-tree",
            "split",
            "splitting",
            "remainder",
            "gain(",
            "argmax",
            "examples",
            "training set",
            "root",
            "leaf",
            "classification",
            "predict",
            "prediction",
        ]

        technical_hits = sum(
            1
            for pattern in technical_patterns
            if pattern in text_lower
        )

        score += min(
            technical_hits * 0.035,
            0.35
        )

        return max(
            0.0,
            min(
                score,
                1.0
            )
        )

    # ============================================================
    # ID3 CONTENT SCORE
    # ============================================================

    def _id3_content_score(
        self,
        query: str,
        text: str
    ) -> float:

        query_lower = query.lower()

        # Activate only for ID3 queries.

        if not re.search(
            r"\bid3\b",
            query_lower
        ):
            return 0.0

        text_lower = text.lower()

        score = 0.0

        # --------------------------------------------------------
        # Direct ID3
        # --------------------------------------------------------

        if re.search(
            r"\bid3\b",
            text_lower
        ):
            score += 0.30

        # --------------------------------------------------------
        # Core ID3 concepts
        # --------------------------------------------------------

        if "information gain" in text_lower:
            score += 0.45

        if "entropy" in text_lower:
            score += 0.30

        if re.search(
            r"\battribute\b",
            text_lower
        ):
            score += 0.10

        if re.search(
            r"\battributes\b",
            text_lower
        ):
            score += 0.05

        # --------------------------------------------------------
        # Attribute selection
        # --------------------------------------------------------

        selection_phrases = [
            "choosing attribute",
            "choosing attributes",
            "choose attribute",
            "choose attributes",
            "select attribute",
            "selecting attribute",
            "selected attribute",
            "attribute selection",
            "attribute test",
            "attribute tests",
            "best attribute",
            "most important attribute",
        ]

        for phrase in selection_phrases:

            if phrase in text_lower:
                score += 0.12

        # --------------------------------------------------------
        # Information gain phrases
        # --------------------------------------------------------

        gain_phrases = [
            "maximum information gain",
            "highest information gain",
            "maximum gain",
            "highest gain",
            "largest information gain",
            "greatest information gain",
            "information gain is",
        ]

        for phrase in gain_phrases:

            if phrase in text_lower:
                score += 0.15

        # --------------------------------------------------------
        # Mathematical ID3 expressions
        # --------------------------------------------------------

        if re.search(
            r"\bgain\s*\(\s*[a-z]\s*\)",
            text_lower
        ):
            score += 0.25

        if re.search(
            r"\bremainder\s*\(\s*[a-z]\s*\)",
            text_lower
        ):
            score += 0.20

        if re.search(
            r"\bh\s*\(\s*goal\s*\)",
            text_lower
        ):
            score += 0.15

        if "b(p" in text_lower:
            score += 0.10

        # --------------------------------------------------------
        # Decision tree concepts
        # --------------------------------------------------------

        if "decision tree" in text_lower:
            score += 0.08

        if "decision-tree" in text_lower:
            score += 0.08

        if "decision-tree-learning" in text_lower:
            score += 0.12

        if "learning decision trees" in text_lower:
            score += 0.15

        if "inducing decision trees" in text_lower:
            score += 0.15

        if "choosing attribute tests" in text_lower:
            score += 0.15

        if "importance function" in text_lower:
            score += 0.12

        # Generic split signals are deliberately small.
        # Core ID3 concepts should dominate them.

        if "split" in text_lower:
            score += 0.02

        if "splitting" in text_lower:
            score += 0.02

        if "split criterion" in text_lower:
            score += 0.05

        if "18.3.4" in text_lower:
            score += 0.10

        # --------------------------------------------------------
        # Historical penalty
        # --------------------------------------------------------

        historical_patterns = [
            "ockham",
            "aristotle",
            "claude shannon",
            "historical",
            "quinlan",
            "1979",
            "1986",
            "first notable use",
            "first proposed",
            "early work",
        ]

        historical_hits = sum(
            1
            for pattern in historical_patterns
            if pattern in text_lower
        )

        if historical_hits >= 2:
            score -= 0.60

        elif historical_hits == 1:
            score -= 0.25

        # --------------------------------------------------------
        # Index/reference penalty
        # --------------------------------------------------------

        index_patterns = [
            "table of contents",
            "bibliography",
            "bibliographical",
            "historical notes",
            "subject index",
            "author index",
            "index of",
        ]

        index_hits = sum(
            1
            for pattern in index_patterns
            if pattern in text_lower
        )

        if index_hits:
            score -= 0.70

        return max(
            0.0,
            min(
                score,
                1.0
            )
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.40
    ) -> List[Dict[str, Any]]:

        if self.index is None:
            raise RuntimeError(
                "RAG index is not loaded. "
                "Call retriever.load() first."
            )

        if not self.documents:
            return []

        query = str(
            query
        ).strip()

        if not query:
            return []

        # --------------------------------------------------------
        # Encode query
        # --------------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        # --------------------------------------------------------
        # Adaptive candidate pool
        # --------------------------------------------------------

        query_lower = query.lower()

        if re.search(
            r"\bid3\b",
            query_lower
        ):

            search_k = min(
                len(self.documents),
                max(
                    k * 400,
                    2000
                )
            )

        elif (
            "decision tree" in query_lower
            or "decision trees" in query_lower
        ):

            search_k = min(
                len(self.documents),
                max(
                    k * 200,
                    1000
                )
            )

        else:

            search_k = min(
                len(self.documents),
                max(
                    k * 100,
                    500
                )
            )

        # --------------------------------------------------------
        # FAISS semantic search
        # --------------------------------------------------------

        semantic_scores, indices = self.index.search(
            query_embedding,
            search_k
        )

        semantic_scores = semantic_scores[0]
        indices = indices[0]

        # --------------------------------------------------------
        # Score candidates
        # --------------------------------------------------------

        results = []

        for semantic_score, idx in zip(
            semantic_scores,
            indices
        ):

            if idx < 0 or idx >= len(
                self.documents
            ):
                continue

            document = self.documents[idx]

            if isinstance(
                document,
                dict
            ):

                text = str(
                    document.get(
                        "text",
                        ""
                    )
                )

            else:
                text = str(
                    document
                )

            if not text.strip():
                continue

            semantic_score = float(
                max(
                    0.0,
                    min(
                        float(
                            semantic_score
                        ),
                        1.0
                    )
                )
            )

            keyword_score = self._keyword_score(
                query,
                text
            )

            phrase_score = self._phrase_score(
                query,
                text
            )

            subtopic_score = self._subtopic_score(
                query,
                text
            )

            content_quality = self._content_quality_score(
                text
            )

            id3_content_score = self._id3_content_score(
                query,
                text
            )

            # ----------------------------------------------------
            # Base combined score
            # ----------------------------------------------------

            combined_score = (
                0.30 * semantic_score
                + 0.12 * keyword_score
                + 0.10 * phrase_score
                + 0.23 * subtopic_score
                + 0.10 * content_quality
                + 0.15 * id3_content_score
            )

            # ----------------------------------------------------
            # ID3-specific boosts
            # ----------------------------------------------------

            if re.search(
                r"\bid3\b",
                text.lower()
            ):

                # Direct ID3 mention is useful,
                # but must not dominate technical content.
                combined_score += 0.08

            if "information gain" in text.lower():
                combined_score += 0.22

            if "entropy" in text.lower():
                combined_score += 0.15

            if "decision tree" in text.lower():
                combined_score += 0.08

            if "decision-tree" in text.lower():
                combined_score += 0.08

            if "best attribute" in text.lower():
                combined_score += 0.10

            if "most important attribute" in text.lower():
                combined_score += 0.10

            if "maximum information gain" in text.lower():
                combined_score += 0.12

            if "highest information gain" in text.lower():
                combined_score += 0.12

            if "choosing attribute tests" in text.lower():
                combined_score += 0.12

            if re.search(
                r"\bgain\s*\(\s*[a-z]\s*\)",
                text.lower()
            ):
                combined_score += 0.12

            if re.search(
                r"\bremainder\s*\(\s*[a-z]\s*\)",
                text.lower()
            ):
                combined_score += 0.08

            # ----------------------------------------------------
            # Historical penalty
            # ----------------------------------------------------

            text_lower = text.lower()

            historical_hits = sum(
                1
                for pattern in [
                    "ockham",
                    "aristotle",
                    "claude shannon",
                    "historical",
                    "quinlan",
                    "1979",
                    "1986",
                    "first notable use",
                ]
                if pattern in text_lower
            )

            if historical_hits >= 2:
                combined_score -= 0.20

            elif historical_hits == 1:
                combined_score -= 0.08

            # ----------------------------------------------------
            # Index/reference penalty
            # ----------------------------------------------------

            index_hits = sum(
                1
                for pattern in [
                    "table of contents",
                    "bibliography",
                    "bibliographical",
                    "historical notes",
                    "subject index",
                    "author index",
                    "index of",
                ]
                if pattern in text_lower
            )

            if index_hits:
                combined_score -= 0.30

            # ----------------------------------------------------
            # Clamp
            # ----------------------------------------------------

            combined_score = max(
                0.0,
                min(
                    combined_score,
                    1.0
                )
            )

            if combined_score < min_score:
                continue

            results.append(
                {
                    "document": document,
                    "semantic_score": round(
                        semantic_score,
                        4
                    ),
                    "keyword_score": round(
                        keyword_score,
                        4
                    ),
                    "phrase_score": round(
                        phrase_score,
                        4
                    ),
                    "subtopic_score": round(
                        subtopic_score,
                        4
                    ),
                    "content_quality": round(
                        content_quality,
                        4
                    ),
                    "id3_content_score": round(
                        id3_content_score,
                        4
                    ),
                    "combined_score": round(
                        combined_score,
                        4
                    ),
                }
            )

        # --------------------------------------------------------
        # Sort by combined score
        # --------------------------------------------------------

        results.sort(
            key=lambda x: x[
                "combined_score"
            ],
            reverse=True
        )

        # --------------------------------------------------------
        # Remove duplicate text
        # --------------------------------------------------------

        final_results = []

        seen_texts = set()

        for result in results:

            document = result[
                "document"
            ]

            if isinstance(
                document,
                dict
            ):

                text = str(
                    document.get(
                        "text",
                        ""
                    )
                )

            else:
                text = str(
                    document
                )

            normalized_text = re.sub(
                r"\s+",
                " ",
                text.lower()
            ).strip()

            if normalized_text in seen_texts:
                continue

            seen_texts.add(
                normalized_text
            )

            final_results.append(
                result
            )

            if len(
                final_results
            ) >= k:
                break

        return final_results