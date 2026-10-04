import pandas as pd

print("=" * 60)
print("QDiff Dataset Diagnostic")
print("=" * 60)

# -------------------------
# TRAINING DATA
# -------------------------

train = pd.read_excel("../qdiff_dataset/train.xlsx")

print("\nTRAINING DATA")
print("-" * 60)

print("Total samples:", len(train))

print("\nDifficulty distribution:")
print(train["Difficulty Level"].value_counts())

print("\nBloom × Difficulty:")
print(
    pd.crosstab(
        train["Bloom's Level"],
        train["Difficulty Level"]
    )
)

# -------------------------
# VALIDATION DATA
# -------------------------

validation = pd.read_excel("../qdiff_dataset/validation.xlsx")

print("\n\nVALIDATION DATA")
print("-" * 60)

print("Total samples:", len(validation))

print("\nDifficulty distribution:")
print(validation["Difficulty Level"].value_counts())

print("\nBloom × Difficulty:")
print(
    pd.crosstab(
        validation["Bloom's Level"],
        validation["Difficulty Level"]
    )
)

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)