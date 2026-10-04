import pandas as pd
import re

FILE = "qdiff_dataset/train.xlsx"

df = pd.read_excel(FILE)

remember = df[df["Bloom's Level"] == "Remember"].copy()


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


remember["normalized"] = remember["Question"].apply(normalize)


patterns = {
    "what is": r"^what\s+is\b",
    "what are": r"^what\s+are\b",
    "what does": r"^what\s+does\b",
    "what do": r"^what\s+do\b",
    "which": r"^which\b",
    "what": r"^what\b",
    "how": r"^how\b",
    "why": r"^why\b",
    "define": r"^define\b",
    "identify": r"^identify\b",
    "name": r"^name\b",
    "list": r"^list\b",
    "state": r"^state\b",
    "recall": r"^recall\b",
    "recognize": r"^recognize\b",
    "label": r"^label\b",
    "write": r"^write\b",
    "select": r"^select\b",
    "match": r"^match\b",
    "outline": r"^outline\b",
    "explain": r"^explain\b",
    "summarize": r"^summarize\b",
    "enumerate": r"^enumerate\b",
    "record": r"^record\b",
    "quote": r"^quote\b",
    "retrieve": r"^retrieve\b",
    "trace": r"^trace\b",
}

print("=" * 80)
print("REMEMBER QUESTION PATTERN ANALYSIS")
print("=" * 80)

print(f"\nTotal Remember training questions: {len(remember)}\n")

for name, pattern in patterns.items():
    count = remember["normalized"].str.match(pattern).sum()
    print(f"{name:12} : {count}")


print("\n" + "=" * 80)
print("NATURAL QUESTION FORMS")
print("=" * 80)

natural_patterns = [
    "what is",
    "what are",
    "what does",
    "what do",
    "which",
    "how",
    "why",
]

for name in natural_patterns:
    pattern = patterns[name]
    matches = remember[
        remember["normalized"].str.match(pattern)
    ]["Question"]

    print(f"\n{name.upper()} QUESTIONS: {len(matches)}")

    for question in matches.tolist():
        print(f"- {question}")


print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)