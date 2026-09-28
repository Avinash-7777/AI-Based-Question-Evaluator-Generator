import pickle

with open("rag_storage/documents.pkl", "rb") as file:
    documents = pickle.load(file)

keyword = "id3"

print(f"Total stored chunks: {len(documents)}")
print(f"\nSearching for: {keyword}\n")

found = 0

for document in documents:

    text = document["text"]

    if keyword.lower() in text.lower():

        found += 1

        print("=" * 70)
        print("Source:", document["source"])
        print("Chunk:", document["chunk_id"])
        print()
        print(text[:1000])
        print()

print("=" * 70)
print("Total matching chunks:", found)