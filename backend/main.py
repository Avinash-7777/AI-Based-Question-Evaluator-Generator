from __future__ import annotations

from feedback_store import save_feedback, get_feedback_summary

import inspect
import logging
import re

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import llm.generator as generator_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm.generator import (
    classify_question_with_llm,
    generate_paper_batch,
    generate_questions,
)

from pydantic import BaseModel
from feedback_learning import (
    list_learning_candidates,
    review_learning_candidate,
    export_approved_training_examples,
)



from qdiff.predictor import predict_question
from rag.retriever import Retriever
from routers.generate_questions import router as generate_questions_router


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qgen-ai")

BASE_DIR = Path(__file__).resolve().parent

BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]



app = FastAPI(
    title="QGen AI",
    description="Reference-grounded AI question generation and evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep your existing question-generation router.
app.include_router(generate_questions_router)


# ============================================================
# REQUEST MODELS
# ============================================================

class FeedbackRequest(BaseModel):
    feature: str = Field(..., min_length=1, max_length=50)
    rating: str
    reason: Optional[str] = Field(default=None, max_length=100)
    comment: Optional[str] = Field(default=None, max_length=2000)
    question: Optional[str] = Field(default=None, max_length=5000)
    predicted_bloom: Optional[str] = None
    corrected_bloom: Optional[str] = None
    predicted_difficulty: Optional[str] = None
    corrected_difficulty: Optional[str] = None
    result_id: Optional[str] = None


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    subtopic: Optional[str] = None
    bloom: str = "Understand"
    difficulty: str = "Medium"
    number_of_questions: int = Field(default=5, ge=1, le=30)


class ClassifyRequest(BaseModel):
    question: str = Field(..., min_length=1)



class PaperRequest(BaseModel):
    subject: str = Field(..., min_length=1)
    scope: str = "all"
    topics: List[str] = Field(default_factory=list)
    pattern: int = 30
    difficulty: str = "balanced"
    number_of_questions: int = Field(default=10, ge=1, le=30)


class ValidateRequest(BaseModel):
    question: Optional[str] = None
    questions: List[str] = Field(default_factory=list)
    context: Optional[str] = None


class AnalyzePaperRequest(BaseModel):
    questions: List[str] = Field(default_factory=list)
    question_paper: Optional[str] = None


# ============================================================
# RAG INITIALIZATION
# ============================================================

retriever: Optional[Retriever] = None


def get_retriever() -> Optional[Retriever]:
    """
    Initialize the retriever and explicitly load its FAISS index.
    Cache it only after loading succeeds.
    """
    global retriever

    if retriever is not None:
        return retriever

    try:
        instance = Retriever()

        load_method = getattr(instance, "load", None)

        if not callable(load_method):
            logger.error(
                "Retriever has no load() method. "
                "Check backend/rag/retriever.py."
            )
            return None

        load_result = load_method()

        if load_result is False:
            logger.error("RAG index loading returned False.")
            return None

        retriever = instance

        logger.info("RAG retriever and index loaded successfully.")
        return retriever

    except Exception:
        logger.exception("Could not load the RAG retriever/index.")
        return None


def retrieve_context(
    query: str,
    k: int = 8,
) -> Dict[str, Any]:
    """
    Retrieve reference material.
    Returns unique source chunks. Missing references remain a
    blocking condition for grounded question generation.
    """
    active_retriever = get_retriever()

    if active_retriever is None:
        logger.error("RAG retriever is unavailable.")
        return {
            "context": "",
            "documents": [],
            "reference_count": 0,
        }

    query = str(query).strip()

    if not query or k <= 0:
        return {
            "context": "",
            "documents": [],
            "reference_count": 0,
        }

    try:
        results = active_retriever.search(
            query,
            k=k,
            min_score=0.20,
        )

        if len(results) < min(3, k):
            logger.info(
                "Only %s reference chunks found; retrying retrieval "
                "with a lower score threshold.",
                len(results),
            )

            retry_results = active_retriever.search(
                query,
                k=min(max(k * 2, 12), 30),
                min_score=0.12,
            )

            results = list(results) + list(retry_results)

    except Exception:
        logger.exception("Reference retrieval failed.")
        return {
            "context": "",
            "documents": [],
            "reference_count": 0,
        }

    documents: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    seen_texts = set()

    for result in results:
        if isinstance(result, dict):
            document = result.get("document", result)
            score = result.get(
                "combined_score",
                result.get("score"),
            )
        else:
            document = getattr(result, "document", result)
            score = getattr(result, "combined_score", None)

        if isinstance(document, dict):
            text = str(document.get("text", "")).strip()
            source = document.get("source", "unknown")
            chunk_id = document.get("chunk_id")
        else:
            text = str(getattr(document, "text", "")).strip()
            source = getattr(document, "source", "unknown")
            chunk_id = getattr(document, "chunk_id", None)

        if not text:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

        if not normalized or normalized in seen_texts:
            continue

        seen_texts.add(normalized)

        documents.append({
            "text": text,
            "source": source,
            "chunk_id": chunk_id,
            "score": score,
        })

        context_parts.append(text)

        if len(documents) >= k:
            break

    context = "\n\n".join(context_parts)

    logger.info(
        "Retrieved %s unique reference chunks for query: %s",
        len(documents),
        query,
    )

    return {
        "context": context,
        "documents": documents,
        "reference_count": len(documents),
    }


# ============================================================
# TEXT / QUALITY HELPERS
# ============================================================

QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\s*)?\d+\s*[.):\-]\s*",
    re.IGNORECASE,
)


def clean_question(value: Any) -> str:
    """Normalize numbering and whitespace without changing meaning."""
    if not isinstance(value, str):
        return ""

    question = value.strip()
    question = QUESTION_PREFIX_RE.sub("", question)
    question = re.sub(r"^\s*[-*•]\s*", "", question)
    question = re.sub(r"\s+", " ", question)

    return question.strip()


def normalize_text(value: str) -> str:
    value = clean_question(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def is_duplicate(
    candidate: str,
    previous_questions: List[str],
    threshold: float = 0.88,
) -> bool:
    candidate_normalized = normalize_text(candidate)

    if not candidate_normalized:
        return True

    for previous in previous_questions:
        previous_normalized = normalize_text(previous)

        if candidate_normalized == previous_normalized:
            return True

        if (
            previous_normalized
            and SequenceMatcher(
                None,
                candidate_normalized,
                previous_normalized,
            ).ratio() >= threshold
        ):
            return True

    return False


def acceptable_question(question: str) -> bool:
    """
    Basic format checks only.
    These checks do not establish factual grounding.
    """
    question = clean_question(question)
    words = question.split()

    if len(words) < 7:
        return False

    if len(words) > 80:
        return False

    if not question.endswith("?"):
        return False

    if re.search(
        r"\b(answer is|the answer is|solution:|correct answer:)\b",
        question,
        re.IGNORECASE,
    ):
        return False

    return True


def call_with_supported_arguments(
    function: Any,
    **kwargs: Any,
) -> Any:
    """
    Pass only supported keyword arguments to existing generator functions.
    This helps preserve compatibility when their signatures differ.
    """
    signature = inspect.signature(function)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)

    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }

    return function(**supported)


# ============================================================
# STRICT GROUNDING
# ============================================================

def verify_question_grounding(
    question: str,
    context: str,
) -> Dict[str, Any]:
    """
    Fail closed if the generator does not provide a grounding validator.
    A question being topically similar to a reference is not enough:
    the generated claim must be explicitly checked against the references.
    """
    question = clean_question(question)

    if not question or not context.strip():
        return {
            "grounded": False,
            "reason": "Question or reference context is empty.",
        }

    validator = getattr(
        generator_module,
        "validate_question_with_llm",
        None,
    )

    if not callable(validator):
        return {
            "grounded": False,
            "reason": (
                "Grounding validator is unavailable. "
                "Generation is blocked to prevent unsupported questions."
            ),
        }

    try:
        result = call_with_supported_arguments(
            validator,
            question=question,
            context=context,
            reference_context=context,
        )

    except Exception:
        logger.exception("Grounding verification failed.")

        return {
            "grounded": False,
            "reason": "Grounding verification failed.",
        }

    if not isinstance(result, dict):
        return {
            "grounded": False,
            "reason": "Grounding validator returned an invalid result.",
        }

    positive = (
        result.get("grounded") is True
        or result.get("is_grounded") is True
        or result.get("supported") is True
        or result.get("is_supported") is True
    )

    explicit_negative = (
        result.get("grounded") is False
        or result.get("is_grounded") is False
        or result.get("supported") is False
        or result.get("is_supported") is False
    )

    if not positive or explicit_negative:
        return {
            "grounded": False,
            "reason": result.get(
                "reason",
                "The question was not explicitly verified against references.",
            ),
        }

    return {
        "grounded": True,
        "reason": result.get("reason", "Verified against references."),
    }


def accept_generated_question(
    question: str,
    context: str,
    previous_questions: List[str],
) -> Dict[str, Any]:
    question = clean_question(question)

    if not acceptable_question(question):
        return {
            "accepted": False,
            "reason": "Question failed the format or quality checks.",
        }

    if is_duplicate(question, previous_questions):
        return {
            "accepted": False,
            "reason": "Duplicate or near-duplicate question.",
        }

    grounding = verify_question_grounding(question, context)

    if not grounding["grounded"]:
        return {
            "accepted": False,
            "reason": grounding["reason"],
        }

    return {
        "accepted": True,
        "question": question,
        "grounded": True,
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def evaluate_question(question: str) -> Dict[str, Any]:
    qdiff_result: Dict[str, Any] = {}
    llm_result: Dict[str, Any] = {}

    # --------------------------------------------------------
    # 1. Get QDiff prediction
    # --------------------------------------------------------
    try:
        prediction = predict_question(question)

        if isinstance(prediction, dict):
            qdiff_result = prediction
        else:
            qdiff_result = {
                "prediction": str(prediction),
            }

    except Exception:
        logger.exception("QDiff classification failed.")
        qdiff_result = {
            "success": False,
            "error": "QDiff classification failed.",
        }

    # --------------------------------------------------------
    # 2. Get LLM prediction
    # --------------------------------------------------------
    try:
        prediction = classify_question_with_llm(question)

        if isinstance(prediction, dict):
            llm_result = prediction
        else:
            llm_result = {
                "prediction": str(prediction),
            }

    except Exception:
        logger.exception("LLM classification failed.")
        llm_result = {
            "success": False,
            "error": "LLM classification failed.",
        }

    # --------------------------------------------------------
    # 3. Extract Bloom level
    # --------------------------------------------------------
    def get_bloom(result: Dict[str, Any]) -> Optional[str]:
        for key in (
            "bloom",
            "bloom_level",
            "Bloom's Level",
            "prediction",
        ):
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    # --------------------------------------------------------
    # 4. Extract difficulty
    # --------------------------------------------------------
    def get_difficulty(
        result: Dict[str, Any],
    ) -> Optional[str]:
        for key in (
            "difficulty",
            "difficulty_level",
            "Difficulty Level",
        ):
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    # --------------------------------------------------------
    # 5. Extract confidence safely
    # --------------------------------------------------------
    def get_confidence(
        result: Dict[str, Any],
        keys: tuple = ("confidence", "bloom_confidence", "score"),
    ) -> Optional[float]:
        for key in keys:
            value = result.get(key)

            try:
                if value is not None:
                    confidence = float(value)

                    if confidence == confidence:  # Exclude NaN
                        return max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                continue

        return None

    qdiff_bloom = get_bloom(qdiff_result)
    llm_bloom = get_bloom(llm_result)

    qdiff_difficulty = get_difficulty(qdiff_result)
    llm_difficulty = get_difficulty(llm_result)

    # QDiff's overall confidence is already calculated by
    # predict_question() from its Bloom and difficulty scores.
    qdiff_confidence = get_confidence(
        qdiff_result,
        ("confidence", "score"),
    )

    # LLM confidence is the value returned by generator.py.
    llm_confidence = get_confidence(
        llm_result,
        ("confidence", "score"),
    )

    # --------------------------------------------------------
    # 6. Compare model predictions
    # --------------------------------------------------------
    bloom_comparable = (
        qdiff_bloom is not None
        and llm_bloom is not None
    )

    difficulty_comparable = (
        qdiff_difficulty is not None
        and llm_difficulty is not None
    )

    bloom_agreement = (
        qdiff_bloom.casefold() == llm_bloom.casefold()
        if bloom_comparable
        else None
    )

    difficulty_agreement = (
        qdiff_difficulty.casefold()
        == llm_difficulty.casefold()
        if difficulty_comparable
        else None
    )

    # Only compare dimensions for which both models
    # returned a prediction.
    agreements = []

    if bloom_comparable:
        agreements.append(bloom_agreement)

    if difficulty_comparable:
        agreements.append(difficulty_agreement)

    llm_qdiff_agreement = (
        round(
            sum(1 for match in agreements if match)
            / len(agreements)
            * 100,
            2,
        )
        if agreements
        else None
    )

    # --------------------------------------------------------
    # 7. Preserve the existing prediction selection rule
    # --------------------------------------------------------
    use_llm = (
        llm_confidence is not None
        and llm_confidence >= 0.60
        and llm_bloom is not None
    )

    chosen = llm_result if use_llm else qdiff_result

    chosen_bloom = get_bloom(chosen)
    chosen_difficulty = get_difficulty(chosen)

    # Keep the existing confidence behavior for compatibility:
    # return the higher available confidence score.
    available_confidences = [
        value
        for value in (qdiff_confidence, llm_confidence)
        if value is not None
    ]

    overall_confidence = (
        max(available_confidences)
        if available_confidences
        else 0.0
    )

    # --------------------------------------------------------
    # 8. Return predictions and agreement metrics
    # --------------------------------------------------------
    return {
        "bloom": chosen_bloom,
        "difficulty": chosen_difficulty,

        # Existing fields retained for frontend compatibility.
        "confidence": round(overall_confidence, 4),
        "classification_source": "LLM" if use_llm else "QDiff",

        # Individual model results.
        "qdiff": qdiff_result,
        "llm": llm_result,

        # Explicit confidence values.
        "qdiff_confidence": (
            round(qdiff_confidence, 4)
            if qdiff_confidence is not None
            else None
        ),
        "llm_confidence": (
            round(llm_confidence, 4)
            if llm_confidence is not None
            else None
        ),

        # Model agreement.
        "bloom_agreement": bloom_agreement,
        "difficulty_agreement": difficulty_agreement,
        "llm_qdiff_agreement": llm_qdiff_agreement,
        "agreement_dimensions_compared": len(agreements),
    }


def paper_blueprint(
    pattern: int,
    number_of_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a paper blueprint using the selected marks pattern and question count."""

    patterns = {
        30: [2, 2, 2, 2, 2, 5, 5, 10],
        50: [2] * 5 + [5] * 4 + [10] * 2,
        60: [2] * 5 + [5] * 5 + [10] + [15],
        100: [2] * 10 + [6] * 5 + [10] * 4,
    }

    if pattern not in patterns:
        raise ValueError("Pattern must be 30, 50, 60, or 100.")

    base_marks = patterns[pattern]

    if number_of_questions is None:
        marks = base_marks.copy()
    else:
        if not 1 <= number_of_questions <= 30:
            raise ValueError(
                "Number of questions must be between 1 and 30."
            )

        # Preserve the original marks sequence and extend it
        # cyclically when more questions are requested.
        marks = [
            base_marks[index % len(base_marks)]
            for index in range(number_of_questions)
        ]

    slots = []

    for index, mark in enumerate(marks):
        if mark <= 2:
            question_type = "short"
        elif mark <= 6:
            question_type = "medium"
        else:
            question_type = "long"

        slots.append({
            "marks": mark,
            "bloom": BLOOM_LEVELS[index % len(BLOOM_LEVELS)],
            "question_type": question_type,
        })

    # The original pattern totals remain valid when no custom
    # question count is provided.
    if number_of_questions is None:
        assert sum(item["marks"] for item in slots) == pattern

    return slots

def question_matches_bloom(
    question: str,
    bloom: str,
) -> bool:
    """
    Lightweight action-word screening. This is not a substitute for
    classification or grounding verification.
    """
    cues = {
        "Remember": (
            r"\bdefine\b",
            r"\blist\b",
            r"\bidentify\b",
            r"\bstate\b",
            r"\bname\b",
        ),
        "Understand": (
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bsummarize\b",
            r"\binterpret\b",
            r"\bwhy\b",
            r"\bhow does\b",
        ),
        "Apply": (
            r"\bapply\b",
            r"\bcalculate\b",
            r"\bcompute\b",
            r"\buse\b",
            r"\bsolve\b",
            r"\bdemonstrate\b",
        ),
        "Analyze": (
            r"\banalyze\b",
            r"\banalyse\b",
            r"\bdifferentiate\b",
            r"\bexamine\b",
            r"\bbreak down\b",
            r"\binvestigate\b",
        ),
        "Evaluate": (
            r"\bevaluate\b",
            r"\bjustify\b",
            r"\bcritique\b",
            r"\bassess\b",
            r"\bdefend\b",
            r"\brecommend\b",
        ),
        "Create": (
            r"\bdesign\b",
            r"\bdevelop\b",
            r"\bconstruct\b",
            r"\bpropose\b",
            r"\bformulate\b",
            r"\bcreate\b",
        ),
    }

    patterns = cues.get(bloom, ())

    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# GENERATION HELPERS
# ============================================================

def extract_question_strings(result: Any) -> List[str]:
    """Normalize common generator response shapes into a list of strings."""
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        items = (
            result.get("questions")
            or result.get("generated_questions")
            or result.get("results")
            or []
        )
    else:
        items = []

    questions: List[str] = []

    for item in items:
        if isinstance(item, str):
            question = clean_question(item)
        elif isinstance(item, dict):
            question = clean_question(
                item.get("question")
                or item.get("text")
                or item.get("content")
                or ""
            )
        else:
            question = ""

        if question:
            questions.append(question)

    return questions


def generate_candidates(
    subject: str,
    context: str,
    number_of_questions: int,
    bloom: str,
    difficulty: str,
    previous_questions: List[str],
    subtopic: Optional[str] = None,
) -> List[str]:
    result = call_with_supported_arguments(
        generate_questions,
        topic=subject,
        subject=subject,
        subtopic=subtopic,
        context=context,
        reference_context=context,
        number_of_questions=number_of_questions,
        bloom=bloom,
        bloom_level=bloom,
        difficulty=difficulty,
        previous_questions=previous_questions,
    )

    return extract_question_strings(result)


def generate_verified_questions(
    subject: str,
    context: str,
    number_of_questions: int,
    bloom: str,
    difficulty: str,
    previous_questions: Optional[List[str]] = None,
    subtopic: Optional[str] = None,
) -> Dict[str, Any]:
    previous = list(previous_questions or [])

    candidates = generate_candidates(
        subject=subject,
        context=context,
        number_of_questions=number_of_questions,
        bloom=bloom,
        difficulty=difficulty,
        previous_questions=previous,
        subtopic=subtopic,
    )

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []

    for candidate in candidates:
        check = accept_generated_question(
            candidate,
            context,
            previous + [item["question"] for item in accepted],
        )

        if not check["accepted"]:
            rejected.append({
                "question": candidate,
                "reason": check["reason"],
            })
            continue

        evaluation = evaluate_question(candidate)

        accepted.append({
            "question": candidate,
            "bloom": evaluation.get("bloom"),
            "difficulty": evaluation.get("difficulty"),
            "confidence": evaluation.get("confidence", 0.0),
            "grounded": True,
            "classification_source": evaluation.get(
                "classification_source"
            ),
        })

    complete = len(accepted) == number_of_questions

    return {
        "success": complete,
        "grounded": bool(accepted) and all(
            item["grounded"] for item in accepted
        ),
        "grounding_status": (
            "verified" if complete else "incomplete_or_blocked"
        ),
        "requested": number_of_questions,
        "generated": len(accepted),
        "missing_count": number_of_questions - len(accepted),
        "questions": accepted,
        "rejected_candidates": rejected,
    }


# ============================================================
# PAPER GENERATION
# ============================================================

def attempt_paper_slot(
    subject: str,
    context: str,
    item: Dict[str, Any],
    difficulty: str,
    previous_questions: List[str],
    selected_topics: List[str],
    all_questions: List[Dict[str, Any]],
    generation_bloom: str,
    generation_type: str,
) -> Optional[Dict[str, Any]]:
    """Generate and validate one strictly grounded paper slot."""
    marks = int(item["marks"])
    requested_bloom = str(item.get("bloom", "")).strip().title()
    actual_bloom = str(
        generation_bloom or requested_bloom
    ).strip().title()

    seen_questions = list(previous_questions)
    seen_questions.extend(
        str(entry.get("question", "")).strip()
        for entry in all_questions
        if entry.get("question")
    )

    unique_seen = list(dict.fromkeys(
        question for question in seen_questions if question
    ))

    excluded_questions = "\n".join(
        f"- {question}" for question in unique_seen
    ) or "- None"

    generation_context = f"""
REFERENCE MATERIAL — THE ONLY SOURCE OF FACTS:

{context}

PAPER SLOT REQUIREMENTS:

- Subject: {subject}
- Marks: {marks}
- Required Bloom level: {actual_bloom}
- Question type: {generation_type}
- Difficulty target: {difficulty}

QUESTIONS ALREADY USED — DO NOT REPEAT OR PARAPHRASE:

{excluded_questions}

STRICT REQUIREMENTS:

1. Generate a genuinely new question for this slot.
2. Prefer a concept or learning objective not already tested
   by the questions listed above.
3. Avoid repeatedly asking for conference names, journal names,
   dates, or definitions when other distinct concepts are
   supported by the references.
4. Use a variety of question styles appropriate to the marks,
   including explanation, application, comparison, and analysis
   when supported by the references.
5. Use only facts and concepts explicitly supported by the
   supplied reference material.
6. Do not invent examples, facts, scenarios, or concepts.
7. Do not reject a valid question solely because its predicted
   Bloom level differs from the requested level.
8. Return fewer questions rather than inventing unsupported content.
9. Identify the central concept tested by every previous
   question before generating the next question.
10. Prefer a different central concept for each new question.
    For example, do not test sensory uncertainty twice when
    other supported AI concepts remain available.
11. Do not use professional society names, conference names,
    journal names, or historical dates as default questions.
    Use them only when they are important learning objectives
    explicitly supported by the references.
12. For 2-mark questions, ask one focused question that can
    reasonably be answered briefly.
13. For 5-mark questions, ask for a developed explanation,
    comparison, or straightforward application when supported.
14. For 10-mark questions, ask a substantial question with
    clearly related components supported by the references.
15. Never add unsupported facts or concepts to improve diversity.
    If no distinct, valid question is available, leave the slot
    missing rather than inventing content.
"""

    try:
        result = call_with_supported_arguments(
            generate_paper_batch,
            subject=subject,
            topic=subject,
            context=generation_context,
            reference_context=context,
            number_of_questions=2,
            bloom=actual_bloom,
            bloom_level=actual_bloom,
            bloom_targets=[actual_bloom],
            difficulty=difficulty,
            question_types=[generation_type],
            question_type=generation_type,
            marks_targets=[marks],
            marks=marks,
            selected_topics=selected_topics,
            previous_questions=unique_seen,
            existing_questions=unique_seen,
            excluded_questions=unique_seen,
            generation_instructions=generation_context,
        )

    except Exception:
        logger.exception("[PAPER] Slot generation failed.")
        return None

    candidates = extract_question_strings(result)

    for candidate in candidates:
        candidate = str(candidate).strip()

        if not candidate or not acceptable_question(candidate):
            logger.info("[PAPER] Rejected: unacceptable question.")
            continue

        if is_duplicate(candidate, unique_seen):
            logger.info("[PAPER] Rejected: duplicate question.")
            continue

        # Verify against actual retrieved references, not the instructions.
        grounding = verify_question_grounding(candidate, context)

        if not grounding.get("grounded", False):
            logger.info("[PAPER] Rejected: grounding check failed.")
            continue

        evaluation = evaluate_question(candidate)

        predicted_bloom = str(
            evaluation.get("bloom") or actual_bloom
        ).strip().title()

        predicted_difficulty = (
            evaluation.get("difficulty") or difficulty
        )

        if predicted_bloom != requested_bloom:
            logger.warning(
                "[PAPER] Bloom mismatch: target=%s, predicted=%s",
                requested_bloom,
                predicted_bloom,
            )

        return {
            "question": candidate,
            "marks": marks,
            "bloom": predicted_bloom,
            "target_bloom": requested_bloom,
            "difficulty": predicted_difficulty,
            "confidence": evaluation.get("confidence", 0.0),
            "question_type": generation_type,
            "grounded": True,
        }

    logger.warning(
        "[PAPER] No acceptable candidate for target Bloom=%s",
        requested_bloom,
    )

    return None

def generate_paper(
    subject: str,
    scope: str,
    topics: List[str],
    pattern: int,
    difficulty: str,
    number_of_questions: int = 10,
) -> Dict[str, Any]:
    """Generate a reference-grounded paper with bounded generation calls."""
    query = subject

    if scope.lower() in {"selected", "selected_topics"} and topics:
        query = f"{subject}. Topics: {', '.join(topics)}"
    elif scope.lower() == "single" and topics:
        query = f"{subject}. Topic: {topics[0]}"

    retrieved = retrieve_context(query, k=20)
    context = retrieved["context"]
    reference_count = retrieved["reference_count"]

    if not context.strip() or reference_count == 0:
        return {
            "success": False,
            "grounded": False,
            "grounding_status": "blocked_no_references",
            "subject": subject,
            "scope": scope,
            "topics": topics,
            "pattern": pattern,
            "difficulty": difficulty,
            "reference_count": 0,
            "total_requested": 0,
            "total_generated": 0,
            "missing_count": 0,
            "completion_rate": 0,
            "questions": [],
            "error": (
                "Generation blocked because no usable reference material "
                "was retrieved."
            ),
        }

    try:
        blueprint = paper_blueprint(
        pattern,
        number_of_questions=number_of_questions,
)

    except ValueError as exc:
        return {
            "success": False,
            "grounded": False,
            "grounding_status": "incomplete_or_blocked",
            "subject": subject,
            "scope": scope,
            "topics": topics,
            "pattern": pattern,
            "difficulty": difficulty,
            "reference_count": reference_count,
            "total_requested": 0,
            "total_generated": 0,
            "missing_count": 0,
            "completion_rate": 0,
            "questions": [],
            "error": str(exc),
        }

    paper_questions: List[Dict[str, Any]] = []
    previous_questions: List[str] = []
    selected_topics = topics if topics else []

    logger.info(
        "[PAPER] Starting bounded generation: %s slots, %s references",
        len(blueprint),
        reference_count,
    )

    for slot_number, item in enumerate(blueprint, start=1):
        target_bloom = item["bloom"]
        question_type = item["question_type"]

        logger.info(
            "[PAPER] Slot %s/%s: marks=%s target_bloom=%s",
            slot_number,
            len(blueprint),
            item["marks"],
            target_bloom,
        )

        question = None

        for slot_attempt in range(1, 3):
            question = attempt_paper_slot(
                subject=subject,
                context=context,
                item=item,
                difficulty=difficulty,
                previous_questions=previous_questions,
                selected_topics=selected_topics,
                all_questions=paper_questions,
                generation_bloom=target_bloom,
                generation_type=question_type,
            )

            if question is not None:
                break

            logger.warning(
                "[PAPER] Slot %s attempt %s failed.",
                slot_number,
                slot_attempt,
            )

        if question is None:
            logger.warning(
                "[PAPER] Slot %s remains missing after retries.",
                slot_number,
            )
            continue

        question["number"] = slot_number
        paper_questions.append(question)
        previous_questions.append(question["question"])

    total_requested = len(blueprint)
    total_generated = len(paper_questions)
    missing_count = total_requested - total_generated
    complete = missing_count == 0

    all_grounded = (
        complete
        and all(
            q.get("grounded") is True
            for q in paper_questions
        )
    )

    return {
        "success": all_grounded,
        "grounded": all_grounded,
        "grounding_status": (
            "verified" if all_grounded else "incomplete_or_blocked"
        ),
        "subject": subject,
        "scope": scope,
        "topics": topics,
        "pattern": pattern,
        "difficulty": difficulty,
        "reference_count": reference_count,
        "total_requested": total_requested,
        "total_generated": total_generated,
        "missing_count": missing_count,
        "completion_rate": (
            round(total_generated / total_requested * 100, 2)
            if total_requested
            else 0
        ),
        "questions": paper_questions,
        "error": (
            None
            if all_grounded
            else (
                f"Could generate only {total_generated} of "
                f"{total_requested} requested questions, or one or more "
                "questions did not pass grounding checks. "
                "The paper is incomplete or blocked."
            )
        ),
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "QGen AI backend is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
@app.get("/api/health")
def health() -> Dict[str, Any]:
    active_retriever = get_retriever()

    return {
        "status": "ok",
        "service": "QGen AI",
        "retriever_initialized": active_retriever is not None,
        "grounding_validator_available": callable(
            getattr(
                generator_module,
                "validate_question_with_llm",
                None,
            )
        ),
    }


@app.post("/api/generate")
def api_generate(request: GenerateRequest) -> Dict[str, Any]:
    query = request.topic

    if request.subtopic:
        query = f"{request.topic}. Subtopic: {request.subtopic}"

    retrieved = retrieve_context(query, k=8)

    if not retrieved["context"].strip():
        return {
            "success": False,
            "grounded": False,
            "grounding_status": "blocked_no_references",
            "topic": request.topic,
            "reference_count": 0,
            "total_requested": request.number_of_questions,
            "total_generated": 0,
            "missing_count": request.number_of_questions,
            "questions": [],
            "error": (
                "Generation blocked because no usable reference material "
                "was retrieved."
            ),
        }

    result = generate_verified_questions(
        subject=request.topic,
        context=retrieved["context"],
        number_of_questions=request.number_of_questions,
        bloom=request.bloom,
        difficulty=request.difficulty,
        subtopic=request.subtopic,
    )

    result.update({
        "topic": request.topic,
        "subtopic": request.subtopic,
        "reference_count": retrieved["reference_count"],
    })

    if not result["success"]:
        result["error"] = (
            f"Could generate and verify only {result['generated']} of "
            f"{result['requested']} requested questions. "
            "The incomplete set is blocked."
        )

    return result


@app.post("/api/classify")
def api_classify(request: ClassifyRequest) -> Dict[str, Any]:
    question = clean_question(request.question)

    return {
        "success": True,
        "question": question,
        **evaluate_question(question),
    }


@app.post("/api/generate-paper")
def api_generate_paper(request: PaperRequest) -> Dict[str, Any]:
    return generate_paper(
        subject=request.subject,
        scope=request.scope,
        topics=request.topics,
        pattern=request.pattern,
        difficulty=request.difficulty,
        number_of_questions=request.number_of_questions,
    )


@app.post("/api/validate")
def api_validate(request: ValidateRequest) -> Dict[str, Any]:
    questions = list(request.questions)

    if request.question and request.question.strip():
        questions.insert(0, request.question.strip())

    questions = [
        clean_question(q)
        for q in questions
        if q.strip()
    ]

    if not questions:
        return {
            "success": False,
            "error": "Provide at least one question to validate.",
            "results": [],
        }

    context = request.context or ""

    if not context.strip():
        retrieved = retrieve_context(" ".join(questions), k=12)
        context = retrieved["context"]
        reference_count = retrieved["reference_count"]
    else:
        reference_count = 0

    if not context.strip():
        return {
            "success": False,
            "grounded": False,
            "grounding_status": "blocked_no_references",
            "reference_count": reference_count,
            "results": [
                {
                    "question": question,
                    "valid": False,
                    "grounded": False,
                    "reason": "No reference context is available.",
                }
                for question in questions
            ],
        }

    results: List[Dict[str, Any]] = []

    for question in questions:
        grounding = verify_question_grounding(question, context)
        evaluation = evaluate_question(question)

        results.append({
            "question": question,
            "valid": grounding["grounded"],
            "grounded": grounding["grounded"],
            "reason": grounding["reason"],
            "bloom": evaluation.get("bloom"),
            "difficulty": evaluation.get("difficulty"),
            "confidence": evaluation.get("confidence", 0.0),
        })

    all_valid = all(item["valid"] for item in results)

    return {
        "success": True,
        "grounded": all_valid,
        "grounding_status": (
            "verified" if all_valid else "contains_unverified_questions"
        ),
        "reference_count": reference_count,
        "total_questions": len(results),
        "valid_count": sum(
            1 for item in results if item["valid"]
        ),
        "invalid_count": sum(
            1 for item in results if not item["valid"]
        ),
        "results": results,
    }


# ============================================================
# ANALYZE QUESTION PAPER
# ============================================================

@app.post("/api/analyze-paper")
def api_analyze_paper(
    request: AnalyzePaperRequest,
) -> Dict[str, Any]:
    questions = list(request.questions)

    if request.question_paper:
        extracted = [
            clean_question(line)
            for line in request.question_paper.splitlines()
        ]

        # Preserve the original behavior for pasted question papers.
        questions.extend(
            q for q in extracted
            if q and q.endswith("?")
        )

    questions = [
        clean_question(q)
        for q in questions
        if q.strip()
    ]

    if not questions:
        return {
            "success": False,
            "error": (
                "Provide questions or paste the question paper text."
            ),
            "results": [],
        }

    results = []

    for question in questions:
        evaluation = evaluate_question(question)

        results.append({
            "question": question,
            "bloom": evaluation.get("bloom"),
            "difficulty": evaluation.get("difficulty"),
            "confidence": evaluation.get("confidence", 0.0),
            "classification_source": evaluation.get(
                "classification_source"
            ),
        })

    # Count Bloom levels and difficulty levels.
    bloom_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}

    for result in results:
        bloom = result.get("bloom") or "Unclassified"
        difficulty = result.get("difficulty") or "Unclassified"

        bloom_counts[bloom] = bloom_counts.get(bloom, 0) + 1

        difficulty_counts[difficulty] = (
            difficulty_counts.get(difficulty, 0) + 1
        )

    total = len(results)

    # NEW: Calculate percentage distribution.
    bloom_percentages = {
        level: round(count / total * 100, 2)
        for level, count in bloom_counts.items()
    }

    difficulty_percentages = {
        level: round(count / total * 100, 2)
        for level, count in difficulty_counts.items()
    }

    # NEW: Determine the dominant overall level.
    # Ties are reported as "Mixed".
    def overall_level(counts: Dict[str, int]) -> str:
        classified = {
            level: count
            for level, count in counts.items()
            if level != "Unclassified"
        }

        if not classified:
            return "Unclassified"

        highest_count = max(classified.values())

        top_levels = [
            level
            for level, count in classified.items()
            if count == highest_count
        ]

        if len(top_levels) > 1:
            return "Mixed"

        return top_levels[0]

    analyzed_questions = sum(
        1
        for result in results
        if result.get("bloom") and result.get("difficulty")
    )

    return {
        "success": True,
        "total_questions": total,
        "analyzed_questions": analyzed_questions,

        # NEW: Overall summary for the frontend.
        "overall_summary": {
            "overall_bloom_level": overall_level(bloom_counts),
            "overall_difficulty": overall_level(difficulty_counts),
        },

        "bloom_distribution": bloom_counts,
        "bloom_percentages": bloom_percentages,

        "difficulty_distribution": difficulty_counts,
        "difficulty_percentages": difficulty_percentages,

        # Individual classifications are preserved.
        "results": results,
    }


# ============================================================
# FEEDBACK
# ============================================================

@app.post("/api/feedback")
def submit_feedback(
    request: FeedbackRequest,
) -> Dict[str, Any]:
    allowed_ratings = {
        "helpful",
        "partly_helpful",
        "not_helpful",
    }

    if request.rating not in allowed_ratings:
        return {
            "success": False,
            "error": "Invalid feedback rating.",
        }

    allowed_bloom = {
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create",
    }

    allowed_difficulty = {
        "Easy",
        "Medium",
        "Hard",
    }

    if (
        request.corrected_bloom is not None
        and request.corrected_bloom not in allowed_bloom
    ):
        return {
            "success": False,
            "error": "Invalid corrected Bloom level.",
        }

    if (
        request.corrected_difficulty is not None
        and request.corrected_difficulty not in allowed_difficulty
    ):
        return {
            "success": False,
            "error": "Invalid corrected difficulty.",
        }

    # NEW: Support both Pydantic v2 and Pydantic v1.
    if hasattr(request, "model_dump"):
        feedback_data = request.model_dump()
    else:
        feedback_data = request.dict()

    feedback_id = save_feedback(feedback_data)

    return {
        "success": True,
        "feedback_id": feedback_id,
        "message": "Feedback saved successfully.",
    }


@app.get("/api/feedback/summary")
def feedback_summary() -> Dict[str, Any]:
    return {
        "success": True,
        **get_feedback_summary(),
    }

class FeedbackReviewRequest(BaseModel):
    approve: bool


@app.get("/api/feedback/learning-candidates")
def get_feedback_learning_candidates():
    try:
        return {
            "success": True,
            "candidates": list_learning_candidates(),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.post("/api/feedback/learning-candidates/{feedback_id}/review")
def review_feedback_learning_candidate(
    feedback_id: int,
    request: FeedbackReviewRequest,
):
    try:
        result = review_learning_candidate(
            feedback_id,
            request.approve,
        )
        return {"success": True, **result}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.post("/api/feedback/export-training-data")
def export_feedback_training_data():
    try:
        result = export_approved_training_examples()
        return {
            "success": True,
            "message": "Approved training examples exported.",
            **result,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
