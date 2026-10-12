
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from qdiff.predictor import predict_question

router = APIRouter(
    prefix="/api",
    tags=["Analyze Question Paper"],
)


class AnalyzePaperRequest(BaseModel):
    questions: List[str] = Field(min_length=1, max_length=100)


BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]


def calculate_distribution(
    results: List[Dict[str, Any]],
    field: str,
    levels: List[str],
) -> Dict[str, Any]:
    counts = {level: 0 for level in levels}

    for result in results:
        value = result.get(field)

        if value in counts:
            counts[value] += 1

    total = sum(counts.values())

    percentages = {
        level: round(count / total * 100, 2) if total else 0.0
        for level, count in counts.items()
    }

    return {
        "counts": counts,
        "percentages": percentages,
        "total_classified": total,
    }


def get_overall_level(distribution: Dict[str, Any]) -> str:
    counts = distribution["counts"]
    total = sum(counts.values())

    if total == 0:
        return "Unavailable"

    highest_count = max(counts.values())
    highest_levels = [
        level
        for level, count in counts.items()
        if count == highest_count
    ]

    if len(highest_levels) > 1:
        return "Mixed"

    return highest_levels[0]


@router.post("/analyze-paper")
def analyze_paper_endpoint(
    request: AnalyzePaperRequest,
) -> Dict[str, Any]:

    results: List[Dict[str, Any]] = []

    # Analyze each non-empty question.
    for original_question in request.questions:
        question = original_question.strip()

        if not question:
            continue

        try:
            prediction = predict_question(question)

            if not isinstance(prediction, dict):
                prediction = {
                    "success": False,
                    "error": "Classifier returned an unexpected result.",
                }

            # Support both common Bloom field names.
            bloom = prediction.get(
                "bloom",
                prediction.get("bloom_level"),
            )

            difficulty = prediction.get("difficulty")

            # Normalize labels to consistent capitalization.
            if isinstance(bloom, str):
                bloom = bloom.strip().title()

            if isinstance(difficulty, str):
                difficulty = difficulty.strip().title()

            prediction_success = prediction.get("success", True)

            result = {
                **prediction,
                "question": question,
                "bloom": bloom,
                "difficulty": difficulty,
                "success": bool(prediction_success),
            }

            # Ensure the normalized Bloom field is also available
            # under the field name used by the classifier.
            if "bloom_level" in prediction:
                result["bloom_level"] = bloom

            results.append(result)

        except Exception as exc:
            results.append({
                "question": question,
                "success": False,
                "error": str(exc),
            })

    successful = [
        result
        for result in results
        if result.get("success", False)
    ]

    bloom_distribution = {
        level: 0 for level in BLOOM_LEVELS
    }
    difficulty_distribution = {
        level: 0 for level in DIFFICULTY_LEVELS
    }

    for result in successful:
        bloom = result.get("bloom") or result.get("bloom_level")
        difficulty = result.get("difficulty")

        if bloom in bloom_distribution:
            bloom_distribution[bloom] += 1

        if difficulty in difficulty_distribution:
            difficulty_distribution[difficulty] += 1

    bloom_total = sum(bloom_distribution.values())
    difficulty_total = sum(difficulty_distribution.values())

    bloom_percentages = {
        level: round(count / bloom_total * 100, 2)
        if bloom_total else 0.0
        for level, count in bloom_distribution.items()
    }

    difficulty_percentages = {
        level: round(count / difficulty_total * 100, 2)
        if difficulty_total else 0.0
        for level, count in difficulty_distribution.items()
    }

    bloom_summary = {
        "counts": bloom_distribution,
        "percentages": bloom_percentages,
        "total_classified": bloom_total,
    }

    difficulty_summary = {
        "counts": difficulty_distribution,
        "percentages": difficulty_percentages,
        "total_classified": difficulty_total,
    }

    overall_bloom_level = get_overall_level(bloom_summary)
    overall_difficulty = get_overall_level(difficulty_summary)

    return {
        "success": bool(results)
        and len(successful) == len(results),

        "total_questions": len(results),
        "analyzed_questions": len(successful),
        "failed_questions": len(results) - len(successful),

        # Preserve the existing distribution fields.
        "bloom_distribution": bloom_distribution,
        "difficulty_distribution": difficulty_distribution,

        # Additional fields for the frontend summary.
        "overall_summary": {
            "overall_bloom_level": overall_bloom_level,
            "overall_difficulty": overall_difficulty,
            "bloom_distribution": bloom_summary,
            "difficulty_distribution": difficulty_summary,
        },

        "results": results,
    }