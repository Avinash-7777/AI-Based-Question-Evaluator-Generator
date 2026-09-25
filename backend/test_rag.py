from rag.retriever import Retriever


retriever = Retriever()

# Build the vector database
retriever.build("data/knowledge_base")


# Ask a question
results = retriever.search(
    "How does a decision tree prevent overfitting?",
    k=3
)


print("\n===== RAG RESULTS =====\n")


for result in results:

    print("Source:", result["document"]["source"])

    print("Distance:", result["distance"])

    print("Text:")
    print(result["document"]["text"])

    print("-" * 60)