from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdiff.predictor import predict_question
from llm.generator import (
    classify_question_with_llm,
    generate_questions,
    refine_question_with_qwen,
)
from rag.retriever import Retriever


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_STORAGE_DIR = BASE_DIR / "rag_storage"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Question Generator API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class GenerateRequest(BaseModel):
    topic: str
    subtopic: str | None = None
    number_of_questions: int = 3
    bloom_level: str | None = None
    difficulty: str | None = None


class ClassifyRequest(BaseModel):
    question: str


# ============================================================
# RAG INITIALIZATION
# ============================================================

retriever = None

try:
    retriever = Retriever()

    print("RAG Retriever object created successfully.")

    retriever.load(str(RAG_STORAGE_DIR))

    print("RAG knowledge base loaded successfully.")
    print(f"RAG storage: {RAG_STORAGE_DIR}")

    try:
        print(f"Total chunks: {len(retriever.documents)}")
    except Exception:
        pass

except Exception as e:
    print(f"WARNING: Could not load RAG retriever: {e}")
    retriever = None


# ============================================================
# RAG SEARCH
# ============================================================

def run_rag(query: str, top_k: int = 5):
    """
    Retrieve relevant chunks from the RAG knowledge base.
    """

    if retriever is None:
        print("RAG retriever is not available.")
        return []

    try:

        results = retriever.search(
            query=query,
            k=top_k,
        )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        print("\n========== RAG DEBUG ==========")
        print("RAG RESULT TYPE:", type(results))
        print("RAG RESULT COUNT:", len(results))

        for i, result in enumerate(results[:5]):

            print(f"\n--- RESULT {i + 1} ---")
            print("TYPE:", type(result))
            print("VALUE:", result)

        print("================================\n")

        print(
            f"RAG retrieved {len(results)} chunks "
            f"for query: {query}"
        )

        return results

    except Exception as e:

        print(f"RAG retrieval error: {e}")

        return []


# ============================================================
# RAG CONTEXT EXTRACTION
# ============================================================

def extract_context(chunks):
    """
    Convert RAG search results into plain text context.

    Current Retriever.search() structure:

    {
        "document": {
            "text": "...actual text...",
            "source": "...",
            "chunk_id": 2444
        },
        "semantic_score": 0.5957,
        "keyword_score": 0.6,
        "phrase_score": 0.12,
        "subtopic_score": 0.75,
        "content_quality": 0.85,
        "id3_content_score": 0.0,
        "combined_score": 1.0
    }
    """

    context_parts = []

    for chunk in chunks:

        # ----------------------------------------------------
        # CASE 1: STRING
        # ----------------------------------------------------

        if isinstance(chunk, str):

            text = chunk.strip()

            if text:
                context_parts.append(text)

            continue

        # ----------------------------------------------------
        # CASE 2: DICTIONARY
        # ----------------------------------------------------

        if isinstance(chunk, dict):

            text = ""

            # ------------------------------------------------
            # CURRENT RAG STRUCTURE
            # ------------------------------------------------

            document = chunk.get("document")

            if isinstance(document, dict):

                text = (
                    document.get("text")
                    or document.get("content")
                    or document.get("chunk")
                    or document.get("page_content")
                    or document.get("raw_text")
                    or ""
                )

            # ------------------------------------------------
            # FALLBACK FOR FLAT STRUCTURES
            # ------------------------------------------------

            if not text:

                text = (
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("chunk")
                    or chunk.get("page_content")
                    or chunk.get("raw_text")
                    or ""
                )

            if isinstance(text, str):

                text = text.strip()

                if text:
                    context_parts.append(text)

            continue

        # ----------------------------------------------------
        # CASE 3: OBJECT WITH ATTRIBUTES
        # ----------------------------------------------------

        for attribute in [
            "text",
            "content",
            "chunk",
            "document",
            "page_content",
            "raw_text",
        ]:

            try:

                value = getattr(
                    chunk,
                    attribute,
                    None,
                )

                # Nested document object
                if attribute == "document":

                    if isinstance(value, dict):

                        text = (
                            value.get("text")
                            or value.get("content")
                            or value.get("chunk")
                            or value.get("page_content")
                            or value.get("raw_text")
                            or ""
                        )

                        if (
                            isinstance(text, str)
                            and text.strip()
                        ):

                            context_parts.append(
                                text.strip()
                            )

                            break

                elif (
                    isinstance(value, str)
                    and value.strip()
                ):

                    context_parts.append(
                        value.strip()
                    )

                    break

            except Exception:
                continue

    return "\n\n".join(context_parts)


# ============================================================
# QDIFF CLASSIFICATION
# ============================================================

def classify_with_qdiff(question: str):

    try:

        result = predict_question(question)

        return {
            "success": True,
            "result": result,
        }

    except Exception as e:

        print(
            f"QDiff classification error: {e}"
        )

        return {
            "success": False,
            "result": None,
            "error": str(e),
        }


# ============================================================
# QWEN CLASSIFICATION
# ============================================================

def classify_with_qwen(question: str):

    try:

        result = classify_question_with_llm(
            question
        )

        return {
            "success": True,
            "result": result,
        }

    except Exception as e:

        print(
            f"Qwen classification error: {e}"
        )

        return {
            "success": False,
            "result": None,
            "error": str(e),
        }


# ============================================================
# COMPARE QDIFF + QWEN
# ============================================================

def compare_predictions(
    question: str,
    qdiff_result,
    qwen_result,
    requested_bloom=None,
    requested_difficulty=None,
):
    """
    Compare QDiff and Qwen predictions.

    QDiff remains the primary classifier unless Qwen has
    strong semantic confidence and QDiff is uncertain.
    """

    qdiff = qdiff_result or {}
    qwen = qwen_result or {}

    qdiff_bloom = qdiff.get(
        "bloom_level"
    )

    qdiff_difficulty = qdiff.get(
        "difficulty"
    )

    qwen_bloom = qwen.get(
        "bloom_level"
    )

    qwen_difficulty = qwen.get(
        "difficulty"
    )

    qdiff_confidence = float(
        qdiff.get("confidence", 0.0) or 0.0
    )

    qwen_confidence = float(
        qwen.get("confidence", 0.0) or 0.0
    )

    final_bloom = qdiff_bloom
    final_difficulty = qdiff_difficulty
    final_confidence = qdiff_confidence

    reason = "QDiff primary prediction."

    # --------------------------------------------------------
    # QWEN HIGH CONFIDENCE + QDIFF LOW CONFIDENCE
    # --------------------------------------------------------

    if (
        qwen_confidence >= 0.85
        and qdiff_confidence < 0.70
        and qwen_bloom == qdiff_bloom
    ):

        final_difficulty = qwen_difficulty
        final_confidence = qwen_confidence

        reason = (
            "Qwen semantic verification used because "
            "QDiff difficulty confidence was low."
        )

    # --------------------------------------------------------
    # FULL AGREEMENT
    # --------------------------------------------------------

    elif (
        qdiff_bloom == qwen_bloom
        and qdiff_difficulty == qwen_difficulty
    ):

        final_bloom = qdiff_bloom
        final_difficulty = qdiff_difficulty

        final_confidence = max(
            qdiff_confidence,
            qwen_confidence,
        )

        reason = "QDiff and Qwen agree."

    # --------------------------------------------------------
    # REQUESTED BLOOM MATCH
    # --------------------------------------------------------

    if requested_bloom:

        if (
            qwen_bloom == requested_bloom
            and qdiff_bloom != requested_bloom
        ):

            if qwen_confidence >= 0.85:

                final_bloom = qwen_bloom

                final_confidence = max(
                    final_confidence,
                    qwen_confidence,
                )

                reason = (
                    "Qwen matched the requested Bloom "
                    "level with high confidence."
                )

    # --------------------------------------------------------
    # REQUESTED DIFFICULTY MATCH
    # --------------------------------------------------------

    if requested_difficulty:

        if (
            qwen_difficulty == requested_difficulty
            and qdiff_difficulty != requested_difficulty
        ):

            if qwen_confidence >= 0.85:

                final_difficulty = qwen_difficulty

                final_confidence = max(
                    final_confidence,
                    qwen_confidence,
                )

                reason = (
                    "Qwen matched the requested difficulty "
                    "with high confidence."
                )

    return {
        "bloom_level": final_bloom,
        "difficulty": final_difficulty,
        "confidence": final_confidence,
        "reason": reason,
        "qdiff": qdiff,
        "qwen": qwen,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    rag_chunks = 0

    if retriever is not None:

        try:
            rag_chunks = len(
                retriever.documents
            )

        except Exception:
            rag_chunks = 0

    return {
        "success": True,
        "status": "running",
        "rag_loaded": retriever is not None,
        "rag_chunks": rag_chunks,
    }


# ============================================================
# GENERATE QUESTIONS
# ============================================================

@app.post("/api/generate")
def generate(request: GenerateRequest):

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    topic = request.topic.strip()

    if not topic:

        return {
            "success": False,
            "error": "Topic is required.",
        }

    number_of_questions = max(
        1,
        min(
            request.number_of_questions,
            10,
        ),
    )

    bloom_level = (
        request.bloom_level
        or "Understand"
    )

    difficulty = (
        request.difficulty
        or "Medium"
    )

    # --------------------------------------------------------
    # BUILD RAG QUERY
    # --------------------------------------------------------

    query_parts = [topic]

    if request.subtopic:

        query_parts.append(
            request.subtopic
        )

    query = " ".join(query_parts)

    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    chunks = run_rag(
        query=query,
        top_k=5,
    )

    # --------------------------------------------------------
    # EXTRACT CONTEXT
    # --------------------------------------------------------

    context = extract_context(
        chunks
    )

    print(
        f"Generation request: "
        f"{number_of_questions} questions | "
        f"Bloom={bloom_level} | "
        f"Difficulty={difficulty}"
    )

    print(
        f"Reference chunks: {len(chunks)}"
    )

    print(
        f"Context length: "
        f"{len(context)} characters"
    )

    # --------------------------------------------------------
    # CONTEXT PREVIEW
    # --------------------------------------------------------

    if context:

        print(
            "\n========== RAG CONTEXT PREVIEW =========="
        )

        print(
            context[:2000]
        )

        if len(context) > 2000:

            print(
                "\n...[context truncated]..."
            )

        print(
            "==========================================\n"
        )

    else:

        print(
            "\nWARNING: RAG returned chunks but "
            "no text was extracted from them.\n"
        )

    # --------------------------------------------------------
    # GENERATE QUESTIONS
    # --------------------------------------------------------

    try:

        generated = generate_questions(
            topic=topic,
            subtopic=request.subtopic,
            context=context,
            number_of_questions=number_of_questions,
            bloom_level=bloom_level,
            difficulty=difficulty,
        )

    except Exception as e:

        print(
            f"Question generation error: {e}"
        )

        return {
            "success": False,
            "error": str(e),
            "reference_chunks": len(
                chunks
            ),
            "context_length": len(
                context
            ),
        }

    # --------------------------------------------------------
    # NORMALIZE GENERATED RESULT
    # --------------------------------------------------------

    if generated is None:

        questions = []

    elif isinstance(
        generated,
        list,
    ):

        questions = generated

    elif isinstance(
        generated,
        dict,
    ):

        questions = generated.get(
            "questions",
            [],
        )

    else:

        try:

            questions = generated.questions

        except Exception:

            questions = []

    print(
        f"Generated questions returned: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # PROCESS QUESTIONS
    # --------------------------------------------------------

    final_questions = []

    for question_item in questions:

        # ----------------------------------------------------
        # EXTRACT QUESTION TEXT
        # ----------------------------------------------------

        if isinstance(
            question_item,
            str,
        ):

            question_text = (
                question_item.strip()
            )

        elif isinstance(
            question_item,
            dict,
        ):

            question_text = (
                question_item.get(
                    "question"
                )
                or question_item.get(
                    "text"
                )
                or question_item.get(
                    "question_text"
                )
                or ""
            ).strip()

        else:

            question_text = (
                getattr(
                    question_item,
                    "question",
                    "",
                )
                or getattr(
                    question_item,
                    "text",
                    "",
                )
                or getattr(
                    question_item,
                    "question_text",
                    "",
                )
                or ""
            ).strip()

        if not question_text:

            continue

        # ----------------------------------------------------
        # QDIFF + QWEN IN PARALLEL
        # ----------------------------------------------------

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            qdiff_future = executor.submit(
                classify_with_qdiff,
                question_text,
            )

            qwen_future = executor.submit(
                classify_with_qwen,
                question_text,
            )

            qdiff_response = (
                qdiff_future.result()
            )

            qwen_response = (
                qwen_future.result()
            )

        qdiff_result = (
            qdiff_response.get(
                "result"
            )
            if qdiff_response.get(
                "success"
            )
            else {}
        )

        qwen_result = (
            qwen_response.get(
                "result"
            )
            if qwen_response.get(
                "success"
            )
            else {}
        )

        # ----------------------------------------------------
        # COMPARE
        # ----------------------------------------------------

        comparison = compare_predictions(
            question=question_text,
            qdiff_result=qdiff_result,
            qwen_result=qwen_result,
            requested_bloom=bloom_level,
            requested_difficulty=difficulty,
        )

        final_bloom = comparison.get(
            "bloom_level"
        )

        final_difficulty = comparison.get(
            "difficulty"
        )

        final_confidence = comparison.get(
            "confidence",
            0.0,
        )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        verified = (
            final_bloom == bloom_level
            and final_difficulty == difficulty
        )

        # ----------------------------------------------------
        # REFINE IF REQUIRED
        # ----------------------------------------------------

        if not verified:

            try:

                refined = (
                    refine_question_with_qwen(
                        question=question_text,
                        requested_bloom=bloom_level,
                        requested_difficulty=difficulty,
                    )
                )

                if refined:

                    if isinstance(
                        refined,
                        str,
                    ):

                        refined_question = (
                            refined.strip()
                        )

                    elif isinstance(
                        refined,
                        dict,
                    ):

                        refined_question = (
                            refined.get(
                                "question"
                            )
                            or refined.get(
                                "text"
                            )
                            or refined.get(
                                "question_text"
                            )
                            or question_text
                        ).strip()

                    else:

                        refined_question = (
                            getattr(
                                refined,
                                "question",
                                "",
                            )
                            or getattr(
                                refined,
                                "text",
                                "",
                            )
                            or question_text
                        ).strip()

                    # ----------------------------------------
                    # RECLASSIFY REFINED QUESTION
                    # ----------------------------------------

                    with ThreadPoolExecutor(
                        max_workers=2
                    ) as executor:

                        qdiff_future = (
                            executor.submit(
                                classify_with_qdiff,
                                refined_question,
                            )
                        )

                        qwen_future = (
                            executor.submit(
                                classify_with_qwen,
                                refined_question,
                            )
                        )

                        refined_qdiff_response = (
                            qdiff_future.result()
                        )

                        refined_qwen_response = (
                            qwen_future.result()
                        )

                    refined_qdiff = (
                        refined_qdiff_response.get(
                            "result"
                        )
                        if refined_qdiff_response.get(
                            "success"
                        )
                        else {}
                    )

                    refined_qwen = (
                        refined_qwen_response.get(
                            "result"
                        )
                        if refined_qwen_response.get(
                            "success"
                        )
                        else {}
                    )

                    refined_comparison = (
                        compare_predictions(
                            question=refined_question,
                            qdiff_result=refined_qdiff,
                            qwen_result=refined_qwen,
                            requested_bloom=bloom_level,
                            requested_difficulty=difficulty,
                        )
                    )

                    question_text = (
                        refined_question
                    )

                    final_bloom = (
                        refined_comparison.get(
                            "bloom_level"
                        )
                    )

                    final_difficulty = (
                        refined_comparison.get(
                            "difficulty"
                        )
                    )

                    final_confidence = (
                        refined_comparison.get(
                            "confidence",
                            0.0,
                        )
                    )

                    verified = (
                        final_bloom == bloom_level
                        and final_difficulty
                        == difficulty
                    )

                    qdiff_result = (
                        refined_qdiff
                    )

                    qwen_result = (
                        refined_qwen
                    )

                    comparison = (
                        refined_comparison
                    )

            except Exception as e:

                print(
                    f"Question refinement error: {e}"
                )

        # ----------------------------------------------------
        # ADD FINAL QUESTION
        # ----------------------------------------------------

        final_questions.append(
            {
                "question": question_text,
                "bloom_level": final_bloom,
                "difficulty": final_difficulty,
                "confidence": final_confidence,
                "verified": verified,
                "qdiff": qdiff_result,
                "qwen": qwen_result,
                "reason": comparison.get(
                    "reason",
                    "",
                ),
            }
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "topic": topic,
        "subtopic": request.subtopic,
        "requested_bloom": bloom_level,
        "requested_difficulty": difficulty,
        "number_requested": number_of_questions,
        "reference_chunks": len(chunks),
        "context_length": len(context),
        "questions": final_questions,
    }


# ============================================================
# CLASSIFY EXISTING QUESTION
# ============================================================

@app.post("/api/classify")
def classify(request: ClassifyRequest):

    question = request.question.strip()

    if not question:

        return {
            "success": False,
            "error": "Question is required.",
        }

    print(
        f"\nClassification request: "
        f"{question}"
    )

    # --------------------------------------------------------
    # QDIFF + QWEN IN PARALLEL
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        qdiff_future = executor.submit(
            classify_with_qdiff,
            question,
        )

        qwen_future = executor.submit(
            classify_with_qwen,
            question,
        )

        qdiff_response = (
            qdiff_future.result()
        )

        qwen_response = (
            qwen_future.result()
        )

    qdiff_result = (
        qdiff_response.get(
            "result"
        )
        if qdiff_response.get(
            "success"
        )
        else {}
    )

    qwen_result = (
        qwen_response.get(
            "result"
        )
        if qwen_response.get(
            "success"
        )
        else {}
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    comparison = compare_predictions(
        question=question,
        qdiff_result=qdiff_result,
        qwen_result=qwen_result,
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "question": question,
        "bloom_level": comparison.get(
            "bloom_level"
        ),
        "difficulty": comparison.get(
            "difficulty"
        ),
        "confidence": comparison.get(
            "confidence",
            0.0,
        ),
        "qdiff": qdiff_result,
        "qwen": qwen_result,
        "reason": comparison.get(
            "reason",
            "",
        ),
    }