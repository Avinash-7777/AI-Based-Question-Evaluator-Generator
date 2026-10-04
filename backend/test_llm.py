from pathlib import Path

from rag.retriever import Retriever
from llm.generator import generate_questions, refine_question_with_qwen
from qdiff.predict import predict_question


# ============================================================
# CONFIGURATION
# ============================================================

topic = "Decision Trees"
subtopic = "ID3 algorithm"

requested_bloom = "Understand"
requested_difficulty = "Easy"
number_of_questions = 2

# Maximum number of Qwen refinement attempts
MAX_REFINEMENT_ATTEMPTS = 2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_STORAGE = BASE_DIR / "rag_storage"


# ============================================================
# LOAD RAG
# ============================================================

print("\n" + "=" * 70)
print("LOADING RAG")
print("=" * 70)

retriever = Retriever()
retriever.load(str(RAG_STORAGE))

print(f"\nRAG knowledge base loaded from: {RAG_STORAGE}")
print("RAG loaded successfully.")


# ============================================================
# RETRIEVE REFERENCE MATERIAL
# ============================================================

query = f"{topic} {subtopic}"

print("\n" + "=" * 70)
print("RAG RETRIEVAL")
print("=" * 70)

print(f"Query: {query}")

retrieved_results = retriever.search(
    query=query,
    k=5,
    min_score=0.40
)


# ============================================================
# CHECK RETRIEVAL
# ============================================================

if not retrieved_results:
    print("\nRelevant reference material not found.")
    raise SystemExit


print(f"\nRetrieved {len(retrieved_results)} chunks.\n")


# ============================================================
# DISPLAY RETRIEVED CHUNKS
# ============================================================

for i, result in enumerate(retrieved_results, start=1):

    document = result["document"]

    print("-" * 70)
    print(f"CHUNK {i}")
    print("-" * 70)

    print(f"Source: {document.get('source')}")
    print(f"Chunk ID: {document.get('chunk_id')}")

    print(
        f"Semantic Score: "
        f"{result.get('semantic_score', 0):.4f}"
    )

    print(
        f"Keyword Score: "
        f"{result.get('keyword_score', 0):.4f}"
    )

    print(
        f"Phrase Score: "
        f"{result.get('phrase_score', 0):.4f}"
    )

    print(
        f"Subtopic Score: "
        f"{result.get('subtopic_score', 0):.4f}"
    )

    print(
        f"Content Quality: "
        f"{result.get('content_quality', 0):.4f}"
    )

    print(
        f"Combined Score: "
        f"{result.get('combined_score', 0):.4f}"
    )

    print("\nText:")

    print(
        document.get("text", "")[:1500]
    )

    print()


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

context = "\n\n".join(
    result["document"]["text"]
    for result in retrieved_results
)


# ============================================================
# GENERATE QUESTIONS WITH QWEN
# ============================================================

print("\n" + "=" * 70)
print("GENERATING QUESTIONS WITH QWEN")
print("=" * 70)

print(f"Topic: {topic}")
print(f"Subtopic: {subtopic}")
print(f"Bloom: {requested_bloom}")
print(f"Difficulty: {requested_difficulty}")
print(f"Number of questions: {number_of_questions}")


questions = generate_questions(
    topic=topic,
    subtopic=subtopic,
    context=context,
    number_of_questions=number_of_questions,
    bloom_level=requested_bloom,
    difficulty=requested_difficulty
)


# ============================================================
# CHECK GENERATED QUESTIONS
# ============================================================

if not questions:

    print("\nNo questions were generated.")
    raise SystemExit


# ============================================================
# INITIAL QUESTIONS
# ============================================================

print("\n" + "=" * 70)
print("INITIAL QWEN QUESTIONS")
print("=" * 70)


for i, question in enumerate(questions, start=1):

    print(f"\nQ{i}: {question}")


# ============================================================
# QDIFF + REFINEMENT LOOP
# ============================================================

final_results = []


for i, question in enumerate(questions, start=1):

    original_question = question
    current_question = question

    print("\n\n" + "=" * 70)
    print(f"QUESTION {i}")
    print("=" * 70)

    print("\nOriginal Question:")
    print(original_question)


    # ========================================================
    # FIRST QDIFF PREDICTION
    # ========================================================

    print("\nRunning QDiff...")

    prediction = predict_question(
        current_question
    )

    detected_bloom = prediction["bloom_level"]
    detected_difficulty = prediction["difficulty"]

    bloom_confidence = prediction["bloom_confidence"]
    difficulty_confidence = prediction["difficulty_confidence"]
    overall_confidence = prediction["confidence"]

    print("\nQDiff Result:")

    print(
        f"Bloom: {detected_bloom}"
    )

    print(
        f"Difficulty: {detected_difficulty}"
    )

    print(
        f"Bloom Confidence: "
        f"{bloom_confidence * 100:.2f}%"
    )

    print(
        f"Difficulty Confidence: "
        f"{difficulty_confidence * 100:.2f}%"
    )

    print(
        f"Overall Confidence: "
        f"{overall_confidence * 100:.2f}%"
    )


    # ========================================================
    # CHECK INITIAL BLOOM + DIFFICULTY
    # ========================================================

    bloom_match = (
        detected_bloom.lower()
        == requested_bloom.lower()
    )

    difficulty_match = (
        detected_difficulty.lower()
        == requested_difficulty.lower()
    )


    # ========================================================
    # QUESTION ALREADY MATCHES
    # ========================================================

    if bloom_match and difficulty_match:

        print("\nVerification: ACCEPTED")

        print(
            "The generated question matches the "
            "requested Bloom level and difficulty."
        )

        final_results.append(
            {
                "question_number": i,

                "original_question": original_question,

                "final_question": current_question,

                "initial_bloom": detected_bloom,

                "initial_difficulty": detected_difficulty,

                "final_bloom": detected_bloom,

                "final_difficulty": detected_difficulty,

                "final_confidence": overall_confidence,

                "status": "ACCEPTED",
            }
        )

        continue


    # ========================================================
    # MISMATCH → REFINEMENT LOOP
    # ========================================================

    print("\nVerification: NEEDS REFINEMENT")

    print(
        "\nMismatch detected between requested "
        "and QDiff classification."
    )

    print(
        f"Requested: "
        f"{requested_bloom} + {requested_difficulty}"
    )

    print(
        f"Detected: "
        f"{detected_bloom} + {detected_difficulty}"
    )


    # ========================================================
    # REFINEMENT LOOP
    # ========================================================

    refinement_success = False

    final_prediction = prediction

    for attempt in range(
        1,
        MAX_REFINEMENT_ATTEMPTS + 1
    ):

        print("\n" + "=" * 70)
        print(
            f"QWEN REFINEMENT ATTEMPT "
            f"{attempt}/{MAX_REFINEMENT_ATTEMPTS}"
        )
        print("=" * 70)

        print(
            "\nSending current question back to Qwen "
            "for refinement..."
        )

        refined_question = refine_question_with_qwen(
            question=current_question,
            topic=topic,
            subtopic=subtopic,
            context=context,
            bloom_level=requested_bloom,
            difficulty=requested_difficulty,
        )


        # ====================================================
        # REFINEMENT FAILED
        # ====================================================

        if not refined_question:

            print(
                "\nQwen refinement attempt failed."
            )

            continue


        # ====================================================
        # SHOW REFINED QUESTION
        # ====================================================

        print("\n" + "-" * 70)
        print(
            f"REFINED QUESTION "
            f"ATTEMPT {attempt}"
        )
        print("-" * 70)

        print(refined_question)


        # ====================================================
        # QDIFF CHECK
        # ====================================================

        print("\nRunning QDiff again...")

        refined_prediction = predict_question(
            refined_question
        )

        refined_bloom = (
            refined_prediction["bloom_level"]
        )

        refined_difficulty = (
            refined_prediction["difficulty"]
        )

        refined_bloom_confidence = (
            refined_prediction["bloom_confidence"]
        )

        refined_difficulty_confidence = (
            refined_prediction["difficulty_confidence"]
        )

        refined_overall_confidence = (
            refined_prediction["confidence"]
        )


        print(
            f"\nQDiff Result "
            f"After Refinement {attempt}:"
        )

        print(
            f"Bloom: {refined_bloom}"
        )

        print(
            f"Difficulty: {refined_difficulty}"
        )

        print(
            f"Bloom Confidence: "
            f"{refined_bloom_confidence * 100:.2f}%"
        )

        print(
            f"Difficulty Confidence: "
            f"{refined_difficulty_confidence * 100:.2f}%"
        )

        print(
            f"Overall Confidence: "
            f"{refined_overall_confidence * 100:.2f}%"
        )


        # ====================================================
        # CHECK REFINED QUESTION
        # ====================================================

        refined_bloom_match = (
            refined_bloom.lower()
            == requested_bloom.lower()
        )

        refined_difficulty_match = (
            refined_difficulty.lower()
            == requested_difficulty.lower()
        )


        # ====================================================
        # ACCEPT REFINED QUESTION
        # ====================================================

        if (
            refined_bloom_match
            and refined_difficulty_match
        ):

            print("\nVerification:")
            print(
                f"ACCEPTED AFTER REFINEMENT "
                f"{attempt}"
            )

            current_question = refined_question
            final_prediction = refined_prediction

            refinement_success = True

            break


        # ====================================================
        # STILL MISMATCHED
        # ====================================================

        print("\nVerification:")
        print(
            f"STILL MISMATCHED AFTER "
            f"REFINEMENT {attempt}"
        )

        print(
            f"Requested: "
            f"{requested_bloom} + {requested_difficulty}"
        )

        print(
            f"Detected: "
            f"{refined_bloom} + {refined_difficulty}"
        )


        # Use the latest refined question as the input
        # for the next refinement attempt.
        current_question = refined_question
        final_prediction = refined_prediction


    # ========================================================
    # DETERMINE FINAL STATUS
    # ========================================================

    if refinement_success:

        final_status = (
            "ACCEPTED_AFTER_REFINEMENT"
        )

    else:

        final_status = (
            "STILL_MISMATCHED"
        )

        print(
            "\nMaximum refinement attempts reached."
        )


    # ========================================================
    # FINAL PREDICTION VALUES
    # ========================================================

    final_bloom = (
        final_prediction["bloom_level"]
    )

    final_difficulty = (
        final_prediction["difficulty"]
    )

    final_overall_confidence = (
        final_prediction["confidence"]
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    final_results.append(
        {
            "question_number": i,

            "original_question": original_question,

            "final_question": current_question,

            "initial_bloom": detected_bloom,

            "initial_difficulty": detected_difficulty,

            "final_bloom": final_bloom,

            "final_difficulty": final_difficulty,

            "final_confidence": final_overall_confidence,

            "status": final_status,
        }
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)


for result in final_results:

    print("\n" + "-" * 70)

    print(
        f"Question {result['question_number']}"
    )

    print("\nOriginal:")

    print(
        result["original_question"]
    )


    if (
        result["original_question"]
        != result["final_question"]
    ):

        print("\nFinal:")

        print(
            result["final_question"]
        )


    print("\nInitial QDiff:")

    print(
        f"Bloom = {result['initial_bloom']}, "
        f"Difficulty = {result['initial_difficulty']}"
    )


    print("\nFinal QDiff:")

    print(
        f"Bloom = {result['final_bloom']}, "
        f"Difficulty = {result['final_difficulty']}"
    )


    print(
        f"\nFinal Confidence: "
        f"{result['final_confidence'] * 100:.2f}%"
    )


    print(
        f"Status: {result['status']}"
    )


# ============================================================
# PIPELINE COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)