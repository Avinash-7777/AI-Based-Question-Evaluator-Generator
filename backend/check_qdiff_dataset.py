import pandas as pd


TRAIN_FILE = "qdiff_dataset/train.xlsx"
VALIDATION_FILE = "qdiff_dataset/validation.xlsx"
TEST_FILE = "qdiff_dataset/test.xlsx"


BLOOM_COLUMN = "Bloom's Level"
DIFFICULTY_COLUMN = "Difficulty Level"
QUESTION_COLUMN = "Question"


def inspect_dataset(file_path, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df = pd.read_excel(file_path)

    print(f"\nFile: {file_path}")
    print(f"Rows: {len(df)}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nBloom distribution:")
    print(df[BLOOM_COLUMN].value_counts())

    print("\nDifficulty distribution:")
    print(df[DIFFICULTY_COLUMN].value_counts())

    print("\nBloom × Difficulty:")
    cross = pd.crosstab(
        df[BLOOM_COLUMN],
        df[DIFFICULTY_COLUMN]
    )

    print(cross)

    print("\nBloom × Difficulty percentages:")

    percentages = pd.crosstab(
        df[BLOOM_COLUMN],
        df[DIFFICULTY_COLUMN],
        normalize="index"
    ) * 100

    print(percentages.round(2))

    return df


# ==============================================================
# LOAD DATASETS
# ==============================================================

train = inspect_dataset(
    TRAIN_FILE,
    "TRAINING DATASET"
)

validation = inspect_dataset(
    VALIDATION_FILE,
    "VALIDATION DATASET"
)

test = inspect_dataset(
    TEST_FILE,
    "TEST DATASET"
)


# ==============================================================
# COMBINE DATASETS
# ==============================================================

print("\n" + "=" * 70)
print("OVERALL DATASET")
print("=" * 70)

all_data = pd.concat(
    [train, validation, test],
    ignore_index=True
)

print(f"\nTotal rows: {len(all_data)}")

print("\nOverall Bloom × Difficulty:")

overall_cross = pd.crosstab(
    all_data[BLOOM_COLUMN],
    all_data[DIFFICULTY_COLUMN]
)

print(overall_cross)

print("\nOverall percentages:")

overall_percentages = pd.crosstab(
    all_data[BLOOM_COLUMN],
    all_data[DIFFICULTY_COLUMN],
    normalize="index"
) * 100

print(overall_percentages.round(2))


# ==============================================================
# ANALYZE QUESTIONS
# ==============================================================

print("\n" + "=" * 70)
print("ANALYZE QUESTIONS")
print("=" * 70)

analyze = all_data[
    all_data[BLOOM_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
    == "analyze"
]

print(f"\nAnalyze questions: {len(analyze)}")

print("\nAnalyze difficulty distribution:")

print(
    analyze[DIFFICULTY_COLUMN].value_counts()
)

print("\nAnalyze difficulty percentages:")

print(
    analyze[DIFFICULTY_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==============================================================
# SAMPLE ANALYZE QUESTIONS
# ==============================================================

print("\n" + "=" * 70)
print("SAMPLE ANALYZE QUESTIONS")
print("=" * 70)

for difficulty in ["Easy", "Medium", "Hard"]:

    subset = analyze[
        analyze[DIFFICULTY_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        == difficulty.lower()
    ]

    print(f"\n--- {difficulty} ---")

    if len(subset) == 0:
        print("No questions found.")
        continue

    for _, row in subset.head(5).iterrows():
        print(f"- {row[QUESTION_COLUMN]}")


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total questions: {len(all_data)}")
print(f"Analyze questions: {len(analyze)}")

print("\nAnalyze → Difficulty:")

for difficulty in ["Easy", "Medium", "Hard"]:

    count = len(
        analyze[
            analyze[DIFFICULTY_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            == difficulty.lower()
        ]
    )

    percentage = (
        count / len(analyze) * 100
        if len(analyze) > 0
        else 0
    )

    print(
        f"{difficulty}: {count} "
        f"({percentage:.2f}%)"
    )

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)