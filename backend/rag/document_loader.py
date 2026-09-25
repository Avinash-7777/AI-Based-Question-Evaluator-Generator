from pathlib import Path


def load_text_files(folder_path: str):
    documents = []

    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "text": text,
            "source": file_path.name
        })

    return documents