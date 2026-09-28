from rag.retriever import Retriever


retriever = Retriever()

retriever.build_from_folder(
    "data/references"
)

retriever.save(
    "rag_storage"
)

print("\nRAG BUILD AND SAVE COMPLETE.")