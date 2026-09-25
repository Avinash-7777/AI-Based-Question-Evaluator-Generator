from llm.generator import generate_questions


questions = generate_questions(
    topic="Machine Learning",
    subtopic="Decision Trees",
    context="""
Decision trees are supervised learning models used
for classification and regression.

A decision tree consists of nodes, branches, and
leaves. Internal nodes represent decisions based
on features, while leaf nodes represent predictions.

Overfitting occurs when a decision tree becomes
too complex and performs poorly on unseen data.

Pruning reduces the complexity of a decision tree
and can help prevent overfitting.
""",
    number_of_questions=3
)


print("\n===== GENERATED QUESTIONS =====\n")

for i, question in enumerate(questions, start=1):
    print(f"{i}. {question}")