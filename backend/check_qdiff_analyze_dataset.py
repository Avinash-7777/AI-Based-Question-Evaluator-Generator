import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

FILES = {
    "TRAIN": "qdiff_dataset/train.xlsx",
    "VALIDATION": "qdiff_dataset/validation.xlsx",
    "TEST": "qdiff_dataset/test.xlsx",
}

QUESTION_COLUMN = "Question"
BLOOM_COLUMN = "Bloom's Level"
DIFFICULTY_COLUMN = "Difficulty Level"


# ============================================================
# ANALYZE DATASET
# ============================================================

for split_name, file_path in FILES.items():

    print()
    print("=" * 80)
    print(f"{split_name} DATASET")
    print("=" * 80)

    df = pd.read_excel(file_path)

    # Clean values
    df[BLOOM_COLUMN] = (
        df[BLOOM_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[DIFFICULTY_COLUMN] = (
        df[DIFFICULTY_COLUMN]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Only Analyze questions
    # --------------------------------------------------------

    analyze_df = df[
        df[BLOOM_COLUMN] == "Analyze"
    ].copy()

    print(
        f"Total Analyze questions: {len(analyze_df)}"
    )

    # --------------------------------------------------------
    # Difficulty distribution
    # --------------------------------------------------------

    print()
    print("Analyze difficulty distribution:")

    print(
        analyze_df[DIFFICULTY_COLUMN]
        .value_counts()
    )

    # --------------------------------------------------------
    # Analyze + Easy
    # --------------------------------------------------------

    easy_df = analyze_df[
        analyze_df[DIFFICULTY_COLUMN] == "Easy"
    ]

    print()
    print("-" * 80)
    print(
        f"ANALYZE + EASY ({len(easy_df)})"
    )
    print("-" * 80)

    for i, question in enumerate(
        easy_df[QUESTION_COLUMN],
        start=1
    ):
        print(f"{i}. {question}")

    # --------------------------------------------------------
    # Analyze + Medium
    # --------------------------------------------------------

    medium_df = analyze_df[
        analyze_df[DIFFICULTY_COLUMN] == "Medium"
    ]

    print()
    print("-" * 80)
    print(
        f"ANALYZE + MEDIUM ({len(medium_df)})"
    )
    print("-" * 80)

    for i, question in enumerate(
        medium_df[QUESTION_COLUMN],
        start=1
    ):
        print(f"{i}. {question}")

    # --------------------------------------------------------
    # Analyze + Hard
    # --------------------------------------------------------

    hard_df = analyze_df[
        analyze_df[DIFFICULTY_COLUMN] == "Hard"
    ]

    print()
    print("-" * 80)
    print(
        f"ANALYZE + HARD ({len(hard_df)})"
    )
    print("-" * 80)

    for i, question in enumerate(
        hard_df[QUESTION_COLUMN],
        start=1
    ):
        print(f"{i}. {question}")

    # --------------------------------------------------------
    # Average question length
    # --------------------------------------------------------

    analyze_df["word_count"] = (
        analyze_df[QUESTION_COLUMN]
        .astype(str)
        .str.split()
        .str.len()
    )

    print()
    print("-" * 80)
    print("AVERAGE QUESTION LENGTH")
    print("-" * 80)

    length_summary = (
        analyze_df
        .groupby(DIFFICULTY_COLUMN)["word_count"]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max"
            ]
        )
    )

    print(length_summary.round(2))

    # --------------------------------------------------------
    # Average character length
    # --------------------------------------------------------

    analyze_df["char_count"] = (
        analyze_df[QUESTION_COLUMN]
        .astype(str)
        .str.len()
    )

    print()
    print("-" * 80)
    print("AVERAGE CHARACTER LENGTH")
    print("-" * 80)

    char_summary = (
        analyze_df
        .groupby(DIFFICULTY_COLUMN)["char_count"]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max"
            ]
        )
    )

    print(char_summary.round(2))


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("ANALYZE DATASET DIAGNOSTIC COMPLETE")
print("=" * 80)