import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    """
    Basic text cleaning.
    """
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# 1. RELEVANCE
# ---------------------------------------------------------

def calculate_relevance(question, query):
    """
    Measures how relevant the generated question is
    to the requested topic/subtopic.

    Uses TF-IDF cosine similarity.
    Returns a score between 0 and 1.
    """

    question = clean_text(question)
    query = clean_text(query)

    if not question or not query:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform([
            question,
            query
        ])

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(float(score), 4)

    except Exception:
        return 0.0


# ---------------------------------------------------------
# 2. GROUNDEDNESS
# ---------------------------------------------------------

def calculate_groundedness(question, context):
    """
    Estimates whether important words from the generated
    question are supported by the retrieved RAG context.

    Returns a score between 0 and 1.
    """

    question = clean_text(question)
    context = clean_text(context)

    if not question or not context:
        return 0.0

    question_words = set(question.split())
    context_words = set(context.split())

    # Remove very common question words
    stop_words = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "does",
        "did",
        "do"
    }

    meaningful_words = {
        word
        for word in question_words
        if word not in stop_words and len(word) > 2
    }

    if not meaningful_words:
        return 0.0

    supported_words = meaningful_words.intersection(context_words)

    score = len(supported_words) / len(meaningful_words)

    return round(float(score), 4)


# ---------------------------------------------------------
# 3. DIVERSITY
# ---------------------------------------------------------

def calculate_diversity(questions):
    """
    Measures how different generated questions are
    from each other.

    Returns a score between 0 and 1.

    1.0 = highly diverse
    0.0 = highly repetitive
    """

    if not questions or len(questions) < 2:
        return 1.0

    cleaned_questions = [
        clean_text(q)
        for q in questions
        if clean_text(q)
    ]

    if len(cleaned_questions) < 2:
        return 1.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            cleaned_questions
        )

        similarities = cosine_similarity(vectors)

        scores = []

        for i in range(len(cleaned_questions)):
            for j in range(i + 1, len(cleaned_questions)):
                scores.append(similarities[i][j])

        if not scores:
            return 1.0

        average_similarity = np.mean(scores)

        diversity = 1.0 - average_similarity

        return round(float(diversity), 4)

    except Exception:
        return 0.0


# ---------------------------------------------------------
# 4. COMPLETE GENERATION EVALUATION
# ---------------------------------------------------------

def evaluate_generation(
    questions,
    topic,
    subtopic,
    context
):
    """
    Evaluate a complete batch of generated questions.
    """

    if not questions:
        return {
            "average_relevance": 0.0,
            "average_groundedness": 0.0,
            "diversity": 0.0
        }

    query = f"{topic} {subtopic}".strip()

    relevance_scores = []
    groundedness_scores = []

    for question in questions:

        relevance = calculate_relevance(
            question,
            query
        )

        groundedness = calculate_groundedness(
            question,
            context
        )

        relevance_scores.append(relevance)
        groundedness_scores.append(groundedness)

    diversity = calculate_diversity(
        questions
    )

    return {
        "average_relevance": round(
            float(np.mean(relevance_scores)),
            4
        ),

        "average_groundedness": round(
            float(np.mean(groundedness_scores)),
            4
        ),

        "diversity": diversity,

        "relevance_scores": relevance_scores,

        "groundedness_scores": groundedness_scores
    }