
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["Generate Question Paper"])


class GeneratePaperRequest(BaseModel):
    subject: str = Field(min_length=1)
    scope: str = "all"
    topics: List[str] = Field(default_factory=list)
    pattern: int = Field(default=30, ge=1)
    difficulty: str = "balanced"


@router.post("/generate-paper")
def generate_paper_endpoint(
    request: GeneratePaperRequest,
) -> Dict[str, Any]:
    # Connect this route to your existing paper-generation service.
    # The service must use RAG references and block incomplete papers.
    try:
        from services.paper_service import generate_question_paper

        result = generate_question_paper(
            subject=request.subject,
            scope=request.scope,
            topics=request.topics,
            pattern=request.pattern,
            difficulty=request.difficulty,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                "Paper service is not connected yet. Move the existing "
                "paper-generation logic into services/paper_service.py."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Paper generation failed: {exc}",
        ) from exc

    return result