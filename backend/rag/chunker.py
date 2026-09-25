def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk.strip())

        start += chunk_size - overlap

    return [chunk for chunk in chunks if chunk]