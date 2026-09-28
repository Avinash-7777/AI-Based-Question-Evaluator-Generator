import re


def clean_text(text: str):
    """
    Clean extracted document text while
    preserving meaningful educational content.
    """

    # Replace multiple spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Reduce excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove leading/trailing spaces
    text = text.strip()

    return text