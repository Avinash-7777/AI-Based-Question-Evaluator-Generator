from qdiff.predict import predict_question


questions = [
    "What is the ID3 algorithm?",
    "What does the ID3 algorithm do?",
    "What does ID3 use to choose an attribute?",
    "What is information gain in ID3?",
    "Name one property of decision trees.",
    "Identify the attribute selected by ID3.",
    "Define the ID3 algorithm.",
    "List one feature of decision trees.",
]


print("=" * 80)
print("QDIFF REMEMBER QUESTION ANALYSIS")
print("=" * 80)


for i, question in enumerate(questions, start=1):

    result = predict_question(question)

    print("\n" + "-" * 80)
    print(f"QUESTION {i}")
    print("-" * 80)

    print(question)

    print()
    print(f"Bloom prediction: {result['bloom_level']}")
    print(f"Difficulty prediction: {result['difficulty']}")

    print()
    print(f"Bloom confidence:      {result['bloom_confidence']:.2%}")
    print(f"Difficulty confidence: {result['difficulty_confidence']:.2%}")
    print(f"Overall confidence:    {result['confidence']:.2%}")


print("\n" + "=" * 80)
print("QDIFF REMEMBER ANALYSIS COMPLETE")
print("=" * 80)