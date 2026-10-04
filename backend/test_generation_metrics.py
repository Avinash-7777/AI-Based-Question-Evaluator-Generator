from evaluation.generation_metrics import (
    calculate_relevance,
    calculate_groundedness,
    calculate_diversity,
    evaluate_generation
)


# ---------------------------------------------------------
# Test questions
# ---------------------------------------------------------

questions = [
    "Who developed the ID3 algorithm and in what year was it introduced?",
    "What crucial idea did ID3 add to decision tree learning regarding attribute selection?",
    "Which system did work on rule induction with ID3 lead to, and who developed that system?"
]


topic = "Artificial Intelligence"

subtopic = "ID3 algorithm"


context = """
ID3 (Quinlan, 1979) added the crucial idea of choosing
the attribute with maximum entropy; it is the basis for
the decision tree algorithm in this chapter.

The ID3 algorithm was developed by J. Ross Quinlan.
His work on rule induction with ID3 eventually led to
the development of the C4.5 system.
"""


# ---------------------------------------------------------
# Individual tests
# ---------------------------------------------------------

print("\n========================================")
print("GENERATION METRICS TEST")
print("========================================")


print("\n1. RELEVANCE")

for question in questions:

    score = calculate_relevance(
        question,
        f"{topic} {subtopic}"
    )

    print(f"\nQuestion: {question}")
    print(f"Relevance: {score}")


print("\n2. GROUNDEDNESS")

for question in questions:

    score = calculate_groundedness(
        question,
        context
    )

    print(f"\nQuestion: {question}")
    print(f"Groundedness: {score}")


print("\n3. DIVERSITY")

diversity = calculate_diversity(
    questions
)

print(f"Diversity: {diversity}")


print("\n4. COMPLETE EVALUATION")

result = evaluate_generation(
    questions=questions,
    topic="Decision Trees",
    subtopic="ID3 algorithm",
    context=context
)

print("\nEvaluation Result:")

for key, value in result.items():
    print(f"{key}: {value}")


print("\n========================================")
print("TEST COMPLETED")
print("========================================")