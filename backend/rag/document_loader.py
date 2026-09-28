from pathlib import Path

from preprocessing.document_loader import load_document
from preprocessing.cleaner import clean_text
from preprocessing.chunker import chunk_text


def process_document(file_path: str):

    path = Path(file_path)

    raw_text = load_document(
        str(path)
    )

    cleaned_text = clean_text(
        raw_text
    )

    chunks = chunk_text(
        cleaned_text
    )

    documents = []

    for index, chunk in enumerate(
        chunks
    ):

        documents.append({

            "text": chunk,

            "source": path.name,

            "chunk_id": index

        })

    return documents