import pandas as pd

FILE = "qdiff_dataset/train.xlsx"

df = pd.read_excel(FILE)

remember = df[df["Bloom's Level"] == "Remember"]

print("=" * 80)
print("REMEMBER QUESTIONS")
print("=" * 80)

print(f"\nTotal Remember questions: {len(remember)}\n")

for i, question in enumerate(remember["Question"], start=1):
    print(f"{i}. {question}")

print("\n" + "=" * 80)
print("REMEMBER QUESTION ANALYSIS")
print("=" * 80)

questions = remember["Question"].astype(str)

patterns = {
    "What is": questions.str.lower().str.startswith("what is"),
    "What are": questions.str.lower().str.startswith("what are"),
    "What does": questions.str.lower().str.startswith("what does"),
    "Which": questions.str.lower().str.startswith("which"),
    "Who": questions.str.lower().str.startswith("who"),
    "Define": questions.str.lower().str.startswith("define"),
    "Name": questions.str.lower().str.startswith("name"),
    "List": questions.str.lower().str.startswith("list"),
    "Identify": questions.str.lower().str.startswith("identify"),
    "State": questions.str.lower().str.startswith("state"),
    "Describe": questions.str.lower().str.startswith("describe"),
    "Explain": questions.str.lower().str.startswith("explain"),
}

for pattern, mask in patterns.items():
    print(f"{pattern:12} : {mask.sum()}")

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)