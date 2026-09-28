from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Question Generator",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RAG IMPORT
# ============================================================

Retriever = None
retrieve_function = None

try:
    from rag.retriever import Retriever

    print("RAG Retriever imported successfully.")

except ImportError as e:

    print("RAG Retriever import failed:", e)

    try:
        from rag.retriever import retrieve as retrieve_function

        print("Fallback RAG retrieve function imported.")

    except ImportError as e2:

        print(
            "Fallback RAG retrieve function import failed:",
            e2
        )


# ============================================================
# LLM IMPORT
# ============================================================

try:

    from llm.generator import generate_questions

    print("LLM generator imported successfully.")

except ImportError as e:

    print("LLM generator import failed:", e)

    generate_questions = None


# ============================================================
# QDIFF IMPORT
# ============================================================

# Your qdiff/predict.py contains:
#
#     predict_question()
#
# NOT:
#
#     predict()

try:

    from qdiff.predict import predict_question

    print("QDiff predictor imported successfully.")

except ImportError as e:

    print("QDiff predictor import failed:", e)

    predict_question = None


# ============================================================
# RAG INITIALIZATION
# ============================================================

retriever = None

try:

    if Retriever is not None:

        retriever = Retriever()

        print("RAG Retriever object created successfully.")

        # ----------------------------------------------------
        # Existing FAISS storage
        # ----------------------------------------------------

        RAG_STORAGE = (
            Path(__file__).resolve().parent
            / "rag_storage"
        )

        if RAG_STORAGE.exists():

            try:

                retriever.load(
                    RAG_STORAGE
                )

                print(
                    "RAG knowledge base loaded successfully."
                )

            except Exception as e:

                print(
                    "RAG knowledge base loading failed:",
                    e
                )

        else:

            print(
                "RAG storage folder not found:",
                RAG_STORAGE
            )

except Exception as e:

    print(
        "RAG initialization failed:",
        e
    )

    retriever = None


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):

    topic: str

    subtopic: str

    number_of_questions: int = 5

    context: Optional[str] = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status": "ok",

        "rag": (
            retriever is not None
            or retrieve_function is not None
        ),

        "llm": (
            generate_questions is not None
        ),

        "qdiff": (
            predict_question is not None
        )
    }


# ============================================================
# RUN RAG
# ============================================================

def run_rag(
    query: str,
    top_k: int = 10
):

    """
    Your Retriever class uses:

        search(query, k=5, min_score=0.40)

    It does NOT use retrieve().
    """

    try:

        # ----------------------------------------------------
        # Existing Retriever class
        # ----------------------------------------------------

        if retriever is not None:

            results = retriever.search(
                query=query,
                k=top_k,
                min_score=0.40
            )

            return results


        # ----------------------------------------------------
        # Optional fallback
        # ----------------------------------------------------

        if retrieve_function is not None:

            results = retrieve_function(
                query,
                top_k=top_k
            )

            return results


        print(
            "RAG is not available."
        )

        return []


    except Exception as e:

        print(
            "RAG retrieval error:",
            e
        )

        return []


# ============================================================
# NORMALIZE RAG RESULT
# ============================================================

def normalize_result(result):

    """
    Converts the output of Retriever.search()
    into a common format.
    """

    # ========================================================
    # Dictionary
    # ========================================================

    if isinstance(result, dict):

        # ----------------------------------------------------
        # Actual Retriever.search() output:
        #
        # {
        #     "document": {
        #         "text": "...",
        #         "source": "...",
        #         "chunk_id": ...
        #     },
        #     "semantic_score": ...,
        #     "keyword_score": ...,
        #     "content_quality": ...,
        #     "combined_score": ...
        # }
        # ----------------------------------------------------

        if "document" in result:

            document = result.get(
                "document",
                {}
            )

            if not isinstance(
                document,
                dict
            ):

                document = {}


            text = document.get(
                "text",
                ""
            )


            return {

                "text": str(
                    text
                ).strip(),

                "score": float(
                    result.get(
                        "combined_score",
                        result.get(
                            "semantic_score",
                            0.0
                        )
                    )
                ),

                "metadata": {

                    "source": document.get(
                        "source",
                        ""
                    ),

                    "chunk_id": document.get(
                        "chunk_id",
                        None
                    ),

                    "semantic_score": float(
                        result.get(
                            "semantic_score",
                            0.0
                        )
                    ),

                    "keyword_score": float(
                        result.get(
                            "keyword_score",
                            0.0
                        )
                    ),

                    "content_quality": float(
                        result.get(
                            "content_quality",
                            0.0
                        )
                    )
                }
            }


        # ----------------------------------------------------
        # Generic dictionary fallback
        # ----------------------------------------------------

        text = (

            result.get(
                "text"
            )

            or

            result.get(
                "content"
            )

            or

            ""
        )


        score = result.get(

            "score",

            result.get(
                "similarity",
                0.0
            )
        )


        metadata = result.get(
            "metadata",
            {}
        )


        return {

            "text": str(
                text
            ).strip(),

            "score": (
                float(score)
                if score is not None
                else 0.0
            ),

            "metadata": metadata
        }


    # ========================================================
    # Tuple / list
    # ========================================================

    if isinstance(
        result,
        (tuple, list)
    ):

        if len(result) >= 2:

            score = result[1]

            if isinstance(
                score,
                (int, float)
            ):

                score = float(score)

            else:

                score = 0.0


            return {

                "text": str(
                    result[0]
                ).strip(),

                "score": score,

                "metadata": {}
            }


        if len(result) == 1:

            return {

                "text": str(
                    result[0]
                ).strip(),

                "score": 0.0,

                "metadata": {}
            }


    # ========================================================
    # String
    # ========================================================

    if isinstance(
        result,
        str
    ):

        return {

            "text": result.strip(),

            "score": 0.0,

            "metadata": {}
        }


    # ========================================================
    # Unknown
    # ========================================================

    return {

        "text": "",

        "score": 0.0,

        "metadata": {}
    }


# ============================================================
# FILTER RELEVANT RESULTS
# ============================================================

def filter_relevant_results(
    results,
    query: str,
    minimum_results: int = 1
):

    """
    Performs a lightweight keyword sanity check
    after the RAG semantic + keyword ranking.
    """

    if not results:

        return []


    # --------------------------------------------------------
    # Query words
    # --------------------------------------------------------

    query_words = {

        word.lower().strip(
            ".,!?;:()[]{}"
        )

        for word in query.split()

        if len(
            word.strip(
                ".,!?;:()[]{}"
            )
        ) >= 3
    }


    filtered = []


    for result in results:

        normalized = normalize_result(
            result
        )

        text = normalized[
            "text"
        ]


        if not text:

            continue


        text_words = {

            word.lower().strip(
                ".,!?;:()[]{}"
            )

            for word in text.split()

            if len(
                word.strip(
                    ".,!?;:()[]{}"
                )
            ) >= 3
        }


        overlap = (
            query_words
            .intersection(
                text_words
            )
        )


        normalized[
            "keyword_overlap"
        ] = len(overlap)


        if overlap:

            filtered.append(
                normalized
            )


    # --------------------------------------------------------
    # Semantic fallback
    # --------------------------------------------------------
    #
    # If exact keyword matching removes everything,
    # keep the original RAG-ranked results.

    if len(filtered) < minimum_results:

        normalized_results = [

            normalize_result(
                result
            )

            for result in results
        ]


        normalized_results = [

            result

            for result in normalized_results

            if result[
                "text"
            ]
        ]


        return normalized_results


    return filtered


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    """
    Combines retrieved chunks into the context
    supplied to the LLM.
    """

    if not results:

        return ""


    context_parts = []


    for index, result in enumerate(
        results,
        start=1
    ):

        normalized = normalize_result(
            result
        )


        text = normalized[
            "text"
        ].strip()


        if not text:

            continue


        metadata = normalized[
            "metadata"
        ]


        source = metadata.get(
            "source",
            "Reference"
        )


        chunk_id = metadata.get(
            "chunk_id",
            ""
        )


        context_parts.append(

            f"[Reference {index} | "
            f"Source: {source} | "
            f"Chunk: {chunk_id}]\n"
            f"{text}"
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# RUN QDIFF
# ============================================================

def run_qdiff(
    question: str
):

    """
    Runs the trained QDiff model.
    """

    if predict_question is None:

        print(
            "QDiff predictor is not available."
        )

        return {

            "bloom_level": None,

            "difficulty": None,

            "bloom_confidence": 0.0,

            "difficulty_confidence": 0.0,

            "confidence": 0.0
        }


    try:

        result = predict_question(
            question
        )


        if not isinstance(
            result,
            dict
        ):

            print(
                "Unexpected QDiff output:",
                result
            )

            return {

                "bloom_level": None,

                "difficulty": None,

                "bloom_confidence": 0.0,

                "difficulty_confidence": 0.0,

                "confidence": 0.0
            }


        return {

            "bloom_level": result.get(
                "bloom_level"
            ),

            "difficulty": result.get(
                "difficulty"
            ),

            "bloom_confidence": float(
                result.get(
                    "bloom_confidence",
                    0.0
                )
            ),

            "difficulty_confidence": float(
                result.get(
                    "difficulty_confidence",
                    0.0
                )
            ),

            "confidence": float(
                result.get(
                    "confidence",
                    0.0
                )
            )
        }


    except Exception as e:

        print(
            "QDiff prediction error:",
            e
        )

        return {

            "bloom_level": None,

            "difficulty": None,

            "bloom_confidence": 0.0,

            "difficulty_confidence": 0.0,

            "confidence": 0.0
        }


# ============================================================
# GENERATE QUESTIONS
# ============================================================

@app.post("/api/generate")
def generate(
    request: GenerateRequest
):

    topic = request.topic.strip()

    subtopic = request.subtopic.strip()

    number_of_questions = (
        request.number_of_questions
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not topic:

        return {

            "success": False,

            "error": "Topic is required.",

            "questions": []
        }


    if not subtopic:

        return {

            "success": False,

            "error": "Subtopic is required.",

            "questions": []
        }


    if number_of_questions < 1:

        return {

            "success": False,

            "error": (
                "Number of questions "
                "must be at least 1."
            ),

            "questions": []
        }


    if number_of_questions > 20:

        return {

            "success": False,

            "error": (
                "Maximum 20 questions "
                "are allowed."
            ),

            "questions": []
        }


    # ========================================================
    # CHECK LLM
    # ========================================================

    if generate_questions is None:

        return {

            "success": False,

            "error": (
                "LLM generator "
                "is not available."
            ),

            "questions": []
        }


    # ========================================================
    # RAG QUERY
    # ========================================================

    # Use the specific subtopic instead of only
    # the broad topic.

    retrieval_query = subtopic


    print()
    print(
        "========================================"
    )

    print(
        "RAG QUERY"
    )

    print(
        "========================================"
    )

    print(
        "Topic:",
        topic
    )

    print(
        "Subtopic:",
        subtopic
    )

    print(
        "Retrieval query:",
        retrieval_query
    )


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    retrieved_results = run_rag(

        retrieval_query,

        top_k=10
    )


    print(
        "Retrieved results:",
        len(
            retrieved_results
        )
    )


    # ========================================================
    # FILTER
    # ========================================================

    filtered_results = (
        filter_relevant_results(

            retrieved_results,

            subtopic,

            minimum_results=1
        )
    )


    print(
        "Relevant results:",
        len(
            filtered_results
        )
    )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = build_context(
        filtered_results
    )


    # ========================================================
    # NO REFERENCE MATERIAL
    # ========================================================

    if not context.strip():

        return {

            "success": False,

            "error": (
                "Relevant reference material "
                "not found for the requested "
                "subtopic."
            ),

            "questions": []
        }


    # ========================================================
    # SHOW CONTEXT
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "RETRIEVED REFERENCE CONTEXT"
    )

    print(
        "========================================"
    )

    print(
        context[:5000]
    )


    # ========================================================
    # LLM GENERATION
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "GENERATING QUESTIONS"
    )

    print(
        "========================================"
    )


    try:

        generated = generate_questions(

            topic=topic,

            subtopic=subtopic,

            context=context,

            number_of_questions=(
                number_of_questions
            )
        )


    except TypeError:

        # ----------------------------------------------------
        # Compatibility fallback
        # ----------------------------------------------------

        try:

            generated = generate_questions(

                topic,

                subtopic,

                context,

                number_of_questions
            )

        except Exception as e:

            print(
                "LLM generation error:",
                e
            )

            return {

                "success": False,

                "error": (
                    f"LLM generation failed: "
                    f"{str(e)}"
                ),

                "questions": []
            }


    except Exception as e:

        print(
            "LLM generation error:",
            e
        )

        return {

            "success": False,

            "error": (
                f"LLM generation failed: "
                f"{str(e)}"
            ),

            "questions": []
        }


    # ========================================================
    # EXTRACT GENERATED QUESTIONS
    # ========================================================

    if hasattr(
        generated,
        "questions"
    ):

        raw_questions = (
            generated.questions
        )


    elif isinstance(
        generated,
        dict
    ):

        raw_questions = (
            generated.get(
                "questions",
                []
            )
        )


    elif isinstance(
        generated,
        list
    ):

        raw_questions = generated


    else:

        raw_questions = []


    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    questions = []


    for item in raw_questions:

        # ----------------------------------------------------
        # Pydantic object
        # ----------------------------------------------------

        if hasattr(
            item,
            "question"
        ):

            question_text = str(
                item.question
            ).strip()


        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        elif isinstance(
            item,
            dict
        ):

            question_text = str(

                item.get(
                    "question",
                    ""
                )

            ).strip()


        # ----------------------------------------------------
        # String
        # ----------------------------------------------------

        elif isinstance(
            item,
            str
        ):

            question_text = item.strip()


        else:

            question_text = ""


        if not question_text:

            continue


        # ====================================================
        # QDIFF
        # ====================================================

        qdiff_result = run_qdiff(
            question_text
        )


        # ====================================================
        # FINAL QUESTION OBJECT
        # ====================================================

        questions.append({

            "question": question_text,

            "bloom_level": (
                qdiff_result[
                    "bloom_level"
                ]
            ),

            "difficulty": (
                qdiff_result[
                    "difficulty"
                ]
            ),

            "bloom_confidence": (
                qdiff_result[
                    "bloom_confidence"
                ]
            ),

            "difficulty_confidence": (
                qdiff_result[
                    "difficulty_confidence"
                ]
            ),

            "confidence": (
                qdiff_result[
                    "confidence"
                ]
            )
        })


    # ========================================================
    # NO QUESTIONS
    # ========================================================

    if not questions:

        return {

            "success": False,

            "error": (
                "The LLM did not generate "
                "valid questions from the "
                "retrieved reference material."
            ),

            "questions": []
        }


    # ========================================================
    # PRINT FINAL QUESTIONS
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "FINAL QUESTIONS"
    )

    print(
        "========================================"
    )


    for index, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"{index}. "
            f"{question['question']}"
        )

        print(
            "   Bloom:",
            question[
                "bloom_level"
            ]
        )

        print(
            "   Difficulty:",
            question[
                "difficulty"
            ]
        )

        print(
            "   Bloom confidence:",
            question[
                "bloom_confidence"
            ]
        )

        print(
            "   Difficulty confidence:",
            question[
                "difficulty_confidence"
            ]
        )

        print(
            "   Overall confidence:",
            question[
                "confidence"
            ]
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success": True,

        "topic": topic,

        "subtopic": subtopic,

        "requested_questions": (
            number_of_questions
        ),

        "generated_questions": len(
            questions
        ),

        "retrieved_chunks": len(
            filtered_results
        ),

        "questions": questions
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "AI Question Generator API is running."
    }