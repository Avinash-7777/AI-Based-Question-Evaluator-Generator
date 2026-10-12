
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from rag.retriever import Retriever

router = APIRouter(prefix="/api", tags=["Validate Questions"])

retriever = Retriever()


class ValidateQuestionsRequest(BaseModel):
    subject: str = Field(min_length=1)
    questions: List[str] = Field(min_length=1, max_length=100)


@router.post("/validate-questions")
def validate_questions_endpoint(
    request: ValidateQuestionsRequest,
) -> Dict[str, Any]:
    from llm.generator import validate_question_with_llm

    results: List[Dict[str, Any]] = []

    for question in request.questions:
        question = question.strip()

        if not question:
            results.append({
                "question": question,
                "valid": False,
                "grounded": False,
                "reason": "Question is empty.",
            })
            continue

        chunks = retriever.search(
            f"{request.subject}: {question}",
            k=5,
        ) or []

        reference_text = []
        for result in chunks:
            document = result.get("document", result)
            text = document.get("text", "") if isinstance(document, dict) else ""
            if text.strip():
                reference_text.append(text.strip())

        context = "\n\n".join(dict.fromkeys(reference_text))

        if not context:
            results.append({
                "question": question,
                "valid": False,
                "grounded": False,
                "reason": "No supporting references found.",
            })
            continue

        try:
            validation = validate_question_with_llm(
                question=question,
                context=context,
            )
        except Exception:
            validation = None

        grounded = (
            isinstance(validation, dict)
            and (
                validation.get("grounded") is True
                or validation.get("is_grounded") is True
                or validation.get("supported") is True
                or validation.get("is_supported") is True
            )
        )

        results.append({
            "question": question,
            "valid": grounded,
            "grounded": grounded,
            "reason": (
                "Reference support verified."
                if grounded
                else "The question could not be verified against the references."
            ),
        })

    all_valid = bool(results) and all(
        result["valid"] for result in results
    )

    return {
        "success": all_valid,
        "grounded": all_valid,
        "total": len(results),
        "validated": sum(result["valid"] for result in results),
        "results": results,
    }