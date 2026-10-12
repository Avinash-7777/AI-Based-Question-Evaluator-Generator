
from pathlib import Path
import logging
import re
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import llm.generator as generator_module
from llm.generator import generate_questions
from rag.retriever import Retriever


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Generate Questions"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
RAG_DIR = BASE_DIR / "rag_storage"

BLOOM_LEVELS = {
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
}

DIFFICULTIES = {"Easy", "Medium", "Hard"}

retriever = Retriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    retriever.load(storage_path=str(RAG_DIR))
    RAG_LOADED = True
    logger.info("Question-generation RAG loaded successfully.")
except Exception:
    RAG_LOADED = False
    logger.exception("Failed to load question-generation RAG.")


class GenerateQuestionsRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    topic: str = Field(min_length=1, max_length=500)
    number_of_questions: int = Field(default=5, ge=1, le=20)
    bloom: str = "Understand"
    difficulty: str = "Medium"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def acceptable_question(question: str) -> bool:
    if not question:
        return False

    words = question.split()

    if not 5 <= len(words) <= 100:
        return False

    forbidden_phrases = [
        "according to the provided context",
        "according to the reference material",
        "based on the given context",
        "the passage states",
        "the document mentions",
    ]

    lowered = question.lower()

    if any(phrase in lowered for phrase in forbidden_phrases):
        return False

    if re.search(r"^\s*(answer|solution)\s*:", lowered):
        return False

    return True


def is_duplicate(question: str, existing: List[str]) -> bool:
    normalized = normalize_text(question)

    for previous in existing:
        old = normalize_text(previous)

        if normalized == old:
            return True

        new_tokens = set(normalized.split())
        old_tokens = set(old.split())

        if not new_tokens or not old_tokens:
            continue

        similarity = len(new_tokens & old_tokens) / len(
            new_tokens | old_tokens
        )

        if similarity >= 0.80:
            return True

    return False


def get_reference_context(
    query: str,
    k: int = 8,
) -> Tuple[str, int]:
    if not RAG_LOADED:
        return "", 0

    results = retriever.search(
        query,
        k=k,
        min_score=0.20,
    )

    chunks: List[str] = []
    seen = set()

    for result in results or []:
        document = result.get("document", result)

        if not isinstance(document, dict):
            continue

        content = str(document.get("text", "")).strip()

        if not content:
            continue

        normalized = normalize_text(content)

        if normalized and normalized not in seen:
            seen.add(normalized)
            chunks.append(content)

    return "\n\n".join(chunks), len(chunks)


def verify_grounding(
    question: str,
    context: str,
) -> Tuple[bool, str]:
    """
    Fail closed: accept a question only if the existing verifier
    explicitly confirms grounding.
    """
    verifier = getattr(
        generator_module,
        "validate_question_with_llm",
        None,
    )

    if not callable(verifier):
        return False, (
            "The grounding verifier is missing from llm/generator.py."
        )

    try:
        result = verifier(question, context)
    except Exception:
        logger.exception("Grounding verification failed.")
        return False, "Grounding verification failed."

    if not isinstance(result, dict):
        return False, "The grounding verifier returned an invalid result."

    grounded = (
        result.get("grounded") is True
        or result.get("is_grounded") is True
        or result.get("supported") is True
        or result.get("is_supported") is True
    )

    explicitly_invalid = (
        result.get("valid") is False
        or result.get("is_valid") is False
    )

    if not grounded or explicitly_invalid:
        return False, "The question was not explicitly verified as grounded."

    return True, "Verified against retrieved reference material."


@router.post("/generate-questions")
def generate_questions_endpoint(
    request: GenerateQuestionsRequest,
) -> Dict[str, Any]:

    subject = request.subject.strip()
    topic = request.topic.strip()
    bloom = request.bloom.strip().title()
    difficulty = request.difficulty.strip().title()

    if bloom not in BLOOM_LEVELS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Bloom level. Choose from: {sorted(BLOOM_LEVELS)}",
        )

    if difficulty not in DIFFICULTIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid difficulty. Choose from: {sorted(DIFFICULTIES)}",
        )

    query = f"{subject}: {topic}"
    context, reference_count = get_reference_context(query)

    if not context.strip() or reference_count == 0:
        return {
            "success": False,
            "grounded": False,
            "grounding_status": "blocked_no_references",
            "subject": subject,
            "topic": topic,
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

    try:
        candidates = generate_questions(
            topic=topic,
            subtopic="",
            context=context,
            number_of_questions=request.number_of_questions,
            bloom=bloom,
            difficulty=difficulty,
        ) or []
    except Exception as exc:
        logger.exception("Question generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Question generation failed. Check the backend logs.",
        ) from exc

    accepted: List[Dict[str, Any]] = []
    seen_questions: List[str] = []
    rejected_count = 0

    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejected_count += 1
            continue

        question = str(candidate.get("question", "")).strip()

        if not acceptable_question(question):
            rejected_count += 1
            continue

        if is_duplicate(question, seen_questions):
            rejected_count += 1
            continue

        grounded, reason = verify_grounding(question, context)

        if not grounded:
            rejected_count += 1
            logger.warning(
                "Rejected ungrounded question: %s | %s",
                question,
                reason,
            )
            continue

        seen_questions.append(question)

        accepted.append({
            **candidate,
            "question": question,
            "grounding_verified": True,
            "grounding_status": "verified",
            "grounding_reason": reason,
            "target_bloom": bloom,
            "target_difficulty": difficulty,
        })

        if len(accepted) >= request.number_of_questions:
            break

    complete = len(accepted) == request.number_of_questions

    return {
        "success": complete,
        "grounded": bool(accepted),
        "grounding_status": (
            "complete"
            if complete
            else "incomplete_or_blocked"
        ),
        "subject": subject,
        "topic": topic,
        "bloom": bloom,
        "difficulty": difficulty,
        "reference_count": reference_count,
        "total_requested": request.number_of_questions,
        "total_generated": len(accepted),
        "missing_count": request.number_of_questions - len(accepted),
        "completion_rate": round(
            len(accepted) / request.number_of_questions,
            2,
        ),
        "rejected_count": rejected_count,
        "questions": accepted,
        "error": (
            None
            if complete
            else (
                "Only verified questions are returned. The requested "
                "number could not be completed using the available "
                "reference material."
            )
        ),
    }