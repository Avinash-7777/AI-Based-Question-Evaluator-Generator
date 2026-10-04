import pandas as pd
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = Path("qdiff_dataset/train.xlsx")
BACKUP_FILE = Path("qdiff_dataset/train_before_remember_augmentation.xlsx")


# ============================================================
# NATURAL REMEMBER QUESTIONS
# ============================================================

new_questions = [
    # --------------------------------------------------------
    # WHAT IS
    # --------------------------------------------------------
    {
        "Question": "What is the ID3 algorithm?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is information gain in decision tree learning?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is a decision tree?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is entropy in decision tree learning?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is a heuristic function in search?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is a Bayesian Network?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is a constraint satisfaction problem?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is a rational agent in Artificial Intelligence?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is a Horn clause?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What is the purpose of the PEAS framework?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },

    # --------------------------------------------------------
    # WHAT ARE
    # --------------------------------------------------------
    {
        "Question": "What are the main components of a decision tree?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the four elements of a PEAS description?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the basic axioms of probability?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the main types of intelligent agents?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the components of a formal search problem?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the common activation functions used in neural networks?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the main types of machine learning?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What are the main components of a Bayesian Belief Network?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },

    # --------------------------------------------------------
    # WHAT DOES
    # --------------------------------------------------------
    {
        "Question": "What does the ID3 algorithm use to select an attribute?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does information gain measure in decision tree learning?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does a heuristic function estimate in search?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does the A* evaluation function calculate?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does a utility function represent in an intelligent agent?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does the unification algorithm find in First-Order Logic?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does a Bayesian Network represent?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What does the entropy formula measure?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },

    # --------------------------------------------------------
    # WHICH
    # --------------------------------------------------------
    {
        "Question": "Which attribute does ID3 select for splitting?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which search strategy uses a FIFO queue?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which search strategy uses a LIFO queue?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which search strategy orders nodes using path cost g(n)?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which search strategy uses the evaluation function f(n) = g(n) + h(n)?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which agent type maintains an internal state?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which activation function is commonly used in a perceptron?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which probability represents the likelihood of evidence given a hypothesis?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },

    # --------------------------------------------------------
    # OTHER NATURAL FACTUAL FORMS
    # --------------------------------------------------------
    {
        "Question": "How is information gain used in ID3?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "How is the path cost g(n) represented in search?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What formula represents the evaluation function of A*?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What term describes an environment where the agent cannot observe the complete state?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "Which term describes the probability of an event before observing evidence?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
    {
        "Question": "What quantity represents the uncertainty of a random variable?",
        "Bloom's Action Word": "Recall",
        "Bloom's Level": "Remember",
        "Difficulty Level": "Easy",
    },
]


# ============================================================
# LOAD DATA
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

df = pd.read_excel(TRAIN_FILE)

required_columns = [
    "Question",
    "Bloom's Action Word",
    "Bloom's Level",
    "Difficulty Level",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# BACKUP
# ============================================================

if not BACKUP_FILE.exists():
    df.to_excel(BACKUP_FILE, index=False)
    print(f"Backup created: {BACKUP_FILE}")
else:
    print(f"Backup already exists: {BACKUP_FILE}")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

existing_questions = set(
    df["Question"]
    .astype(str)
    .str.strip()
    .str.lower()
)

unique_new_questions = []

for item in new_questions:

    normalized = item["Question"].strip().lower()

    if normalized not in existing_questions:
        unique_new_questions.append(item)
        existing_questions.add(normalized)


# ============================================================
# APPEND
# ============================================================

new_df = pd.DataFrame(unique_new_questions)

df_augmented = pd.concat(
    [df, new_df],
    ignore_index=True
)

df_augmented.to_excel(
    TRAIN_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("QDIFF REMEMBER DATASET AUGMENTATION")
print("=" * 80)

print(f"Original training rows : {len(df)}")
print(f"New questions prepared : {len(new_questions)}")
print(f"New questions added    : {len(unique_new_questions)}")
print(f"Final training rows   : {len(df_augmented)}")

print()
print("Bloom distribution:")
print(df_augmented["Bloom's Level"].value_counts())

print()
print("Remember distribution:")
print(
    df_augmented[
        df_augmented["Bloom's Level"] == "Remember"
    ]["Difficulty Level"].value_counts()
)

print()
print("=" * 80)
print("AUGMENTATION COMPLETE")
print("=" * 80)