from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.retriever import Retriever
from llm.generator import generate_questions
from qdiff.predict import predict_question


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Question Generator API",
    description=(
        "Backend for AI-Based Question "
        "Evaluator and Generator"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RAG
# ============================================================

print("\nLoading RAG system...")

retriever = Retriever()

retriever.build(
    "data/knowledge_base"
)

print("RAG system loaded successfully.")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Question Generator API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "rag": "ready",
        "qdiff": "ready"
    }


# ============================================================
# GENERATE QUESTIONS
# ============================================================

@app.post("/api/generate")
def generate(data: dict):

    # --------------------------------------------------------
    # Get input
    # --------------------------------------------------------

    topic = data.get(
        "topic",
        ""
    ).strip()

    subtopic = data.get(
        "subtopic",
        ""
    ).strip()

    number_of_questions = data.get(
        "number_of_questions",
        5
    )


    # --------------------------------------------------------
    # Validate topic
    # --------------------------------------------------------

    if not topic:

        return {
            "error": "Topic is required."
        }


    # --------------------------------------------------------
    # Create RAG query
    # --------------------------------------------------------

    query = topic

    if subtopic:

        query = (
            f"{topic} {subtopic}"
        )


    print("\n" + "=" * 60)

    print("GENERATING QUESTIONS")

    print("=" * 60)

    print("Topic:", topic)
    print("Subtopic:", subtopic)
    print(
        "Number of questions:",
        number_of_questions
    )


    # ========================================================
    # STEP 1 — RAG
    # ========================================================

    print("\n[1/3] Searching knowledge base...")

    results = retriever.search(
        query,
        k=3
    )


    context_parts = []

    sources = []


    for result in results:

        context_parts.append(
            result["document"]["text"]
        )

        source = result[
            "document"
        ]["source"]

        if source not in sources:

            sources.append(
                source
            )


    context = "\n\n".join(
        context_parts
    )


    print(
        "Retrieved chunks:",
        len(results)
    )


    # ========================================================
    # STEP 2 — LLM
    # ========================================================

    print(
        "\n[2/3] Generating questions..."
    )


    questions = generate_questions(

        topic=topic,

        subtopic=subtopic,

        context=context,

        number_of_questions=(
            number_of_questions
        )
    )


    print(
        "Generated questions:",
        len(questions)
    )


    # ========================================================
    # STEP 3 — QDIFF
    # ========================================================

    print(
        "\n[3/3] Classifying questions with QDiff..."
    )


    formatted_questions = []


    for index, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"Classifying question {index}..."
        )


        prediction = predict_question(
            question
        )


        formatted_questions.append({

            "id": index,

            "question": question,

            "bloom_level": prediction[
                "bloom_level"
            ],

            "difficulty": prediction[
                "difficulty"
            ],

            "bloom_confidence": prediction[
                "bloom_confidence"
            ],

            "difficulty_confidence": prediction[
                "difficulty_confidence"
            ],

            "confidence": prediction[
                "confidence"
            ]

        })


    # ========================================================
    # RESPONSE
    # ========================================================

    print(
        "\nGeneration completed successfully."
    )


    return {

        "topic": topic,

        "subtopic": subtopic,

        "number_of_questions": (
            len(formatted_questions)
        ),

        "sources": sources,

        "questions": formatted_questions

    }