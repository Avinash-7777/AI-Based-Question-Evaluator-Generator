from generation_metrics import evaluate_generation


# ============================================================
# TEST QUESTIONS
# ============================================================

questions = [
    "Analyze how information gain influences the selection of attributes in the ID3 algorithm?",
    "Compare the information gain of two attributes in a decision tree and investigate under what conditions the information gain remains zero?",
]


# ============================================================
# TEST QUERY
# ============================================================

query = "Decision Trees ID3 algorithm"


# ============================================================
# TEST CONTEXT
# ============================================================

context = """
ID3 is a decision tree learning algorithm that selects an attribute
based on information gain. Information gain measures the expected
reduction in entropy resulting from splitting the examples according
to an attribute. The attribute with the highest information gain
can be selected as the splitting attribute. An attribute with zero
information gain does not reduce the uncertainty in the examples.
"""


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATION METRICS TEST")
print("=" * 70)

results = evaluate_generation(
    questions=questions,
    topic="Decision Trees",
    subtopic="ID3 algorithm",
    context=context
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

for key, value in results.items():

    print(
        f"{key}: {value}"
    )


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
