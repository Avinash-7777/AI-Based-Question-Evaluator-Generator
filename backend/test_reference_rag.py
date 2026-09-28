from rag.retriever import Retriever


retriever = Retriever()

# Load the already-built RAG knowledge base
retriever.load(
    "rag_storage"
)


query = input(
    "\nEnter topic/subtopic to search: "
)


results = retriever.search(
    query,
    k=5,
    min_score=0.40
)


print(
    "\n\n===== RETRIEVED REFERENCE CHUNKS =====\n"
)


if not results:

    print(
        "No sufficiently relevant reference "
        "material found."
    )

else:

    for index, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        print(
            f"----- Result {index} -----"
        )

        print(
            "Source:",
            document["source"]
        )

        print(
            "Chunk:",
            document["chunk_id"]
        )

        print(
            "Semantic similarity:",
            f"{result['semantic_score']:.4f}"
        )

        print(
            "Keyword score:",
            f"{result['keyword_score']:.4f}"
        )

        print(
            "Combined score:",
            f"{result['combined_score']:.4f}"
        )

        print("\nText:")

        print(
            document["text"]
        )

        print(
            "\n" + "-" * 70 + "\n"
        )