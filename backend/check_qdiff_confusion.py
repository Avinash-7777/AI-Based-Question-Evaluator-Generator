import pandas as pd
import torch

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from qdiff.model import QDiffModel
from qdiff.labels import BLOOM_TO_ID, DIFFICULTY_TO_ID


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MODEL_PATH = "qdiff_model/qdiff_model.pt"

VAL_FILE = "qdiff_dataset/validation.xlsx"

BATCH_SIZE = 8
MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("QDiff device:", DEVICE)


# ============================================================
# LABEL MAPPINGS
# ============================================================

ID_TO_BLOOM = {
    value: key
    for key, value in BLOOM_TO_ID.items()
}

ID_TO_DIFFICULTY = {
    value: key
    for key, value in DIFFICULTY_TO_ID.items()
}


# ============================================================
# DATASET
# ============================================================

class QuestionDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        question = str(row["Question"])

        encoding = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        bloom_label = BLOOM_TO_ID[
            str(row["Bloom's Level"]).strip()
        ]

        difficulty_label = DIFFICULTY_TO_ID[
            str(row["Difficulty Level"]).strip()
        ]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),

            "attention_mask": encoding[
                "attention_mask"
            ].squeeze(0),

            "bloom_label": bloom_label,

            "difficulty_label": difficulty_label,

            "question": question
        }


# ============================================================
# LOAD DATA
# ============================================================

val_df = pd.read_excel(VAL_FILE)

print("Validation samples:", len(val_df))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET / DATALOADER
# ============================================================

dataset = QuestionDataset(
    val_df,
    tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = QDiffModel(
    model_name=MODEL_NAME
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("QDiff model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

all_bloom_true = []
all_bloom_pred = []

all_difficulty_true = []
all_difficulty_pred = []

all_questions = []


with torch.no_grad():

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)


        bloom_logits, difficulty_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        bloom_predictions = torch.argmax(
            bloom_logits,
            dim=1
        ).cpu().tolist()


        difficulty_predictions = torch.argmax(
            difficulty_logits,
            dim=1
        ).cpu().tolist()


        # Convert true labels to normal Python integers

        bloom_true = [
            int(x)
            for x in batch["bloom_label"]
        ]

        difficulty_true = [
            int(x)
            for x in batch["difficulty_label"]
        ]


        all_bloom_true.extend(
            bloom_true
        )

        all_bloom_pred.extend(
            bloom_predictions
        )

        all_difficulty_true.extend(
            difficulty_true
        )

        all_difficulty_pred.extend(
            difficulty_predictions
        )

        all_questions.extend(
            batch["question"]
        )


# ============================================================
# BLOOM CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("BLOOM CONFUSION MATRIX")
print("=" * 70)

bloom_labels = list(
    range(len(ID_TO_BLOOM))
)

bloom_matrix = confusion_matrix(
    all_bloom_true,
    all_bloom_pred,
    labels=bloom_labels
)

bloom_names = [
    ID_TO_BLOOM[i]
    for i in bloom_labels
]

bloom_matrix_df = pd.DataFrame(
    bloom_matrix,
    index=[
        f"Actual {name}"
        for name in bloom_names
    ],
    columns=[
        f"Pred {name}"
        for name in bloom_names
    ]
)

print(bloom_matrix_df)


# ============================================================
# DIFFICULTY CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("DIFFICULTY CONFUSION MATRIX")
print("=" * 70)

difficulty_labels = list(
    range(len(ID_TO_DIFFICULTY))
)

difficulty_matrix = confusion_matrix(
    all_difficulty_true,
    all_difficulty_pred,
    labels=difficulty_labels
)

difficulty_names = [
    ID_TO_DIFFICULTY[i]
    for i in difficulty_labels
]

difficulty_matrix_df = pd.DataFrame(
    difficulty_matrix,
    index=[
        f"Actual {name}"
        for name in difficulty_names
    ],
    columns=[
        f"Pred {name}"
        for name in difficulty_names
    ]
)

print(difficulty_matrix_df)


# ============================================================
# DIFFICULTY CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("DIFFICULTY CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_difficulty_true,
        all_difficulty_pred,
        labels=difficulty_labels,
        target_names=difficulty_names,
        digits=4
    )
)


# ============================================================
# ANALYZE QUESTIONS ONLY
# ============================================================

print()
print("=" * 70)
print("ANALYZE QUESTIONS: DIFFICULTY PREDICTIONS")
print("=" * 70)

analyze_id = BLOOM_TO_ID["Analyze"]

analyze_rows = []


for i in range(len(all_questions)):

    if all_bloom_true[i] == analyze_id:

        actual_difficulty = ID_TO_DIFFICULTY[
            int(all_difficulty_true[i])
        ]

        predicted_difficulty = ID_TO_DIFFICULTY[
            int(all_difficulty_pred[i])
        ]

        analyze_rows.append({
            "Question": all_questions[i],
            "Actual": actual_difficulty,
            "Predicted": predicted_difficulty
        })


analyze_df = pd.DataFrame(
    analyze_rows
)


# ============================================================
# ANALYZE CONFUSION MATRIX
# ============================================================

print()
print("Analyze confusion matrix:")

analyze_matrix = pd.crosstab(
    analyze_df["Actual"],
    analyze_df["Predicted"]
)

print(analyze_matrix)


# ============================================================
# ANALYZE → MEDIUM ERRORS
# ============================================================

print()
print("=" * 70)
print("ACTUAL MEDIUM ANALYZE QUESTIONS PREDICTED AS HARD")
print("=" * 70)

medium_to_hard = analyze_df[
    (analyze_df["Actual"] == "Medium") &
    (analyze_df["Predicted"] == "Hard")
]

print(
    "Count:",
    len(medium_to_hard)
)


for _, row in medium_to_hard.head(20).iterrows():

    print()
    print("Question:", row["Question"])
    print("Actual:", row["Actual"])
    print("Predicted:", row["Predicted"])


# ============================================================
# ANALYZE → HARD ERRORS
# ============================================================

print()
print("=" * 70)
print("ACTUAL HARD ANALYZE QUESTIONS PREDICTED AS MEDIUM")
print("=" * 70)

hard_to_medium = analyze_df[
    (analyze_df["Actual"] == "Hard") &
    (analyze_df["Predicted"] == "Medium")
]

print(
    "Count:",
    len(hard_to_medium)
)


for _, row in hard_to_medium.head(20).iterrows():

    print()
    print("Question:", row["Question"])
    print("Actual:", row["Actual"])
    print("Predicted:", row["Predicted"])


# ============================================================
# ANALYZE → EASY ERRORS
# ============================================================

print()
print("=" * 70)
print("ACTUAL MEDIUM ANALYZE QUESTIONS PREDICTED AS EASY")
print("=" * 70)

medium_to_easy = analyze_df[
    (analyze_df["Actual"] == "Medium") &
    (analyze_df["Predicted"] == "Easy")
]

print(
    "Count:",
    len(medium_to_easy)
)


for _, row in medium_to_easy.head(20).iterrows():

    print()
    print("Question:", row["Question"])
    print("Actual:", row["Actual"])
    print("Predicted:", row["Predicted"])


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)


total = len(
    all_difficulty_true
)

correct = sum(
    true == pred
    for true, pred in zip(
        all_difficulty_true,
        all_difficulty_pred
    )
)


print(
    f"Overall difficulty accuracy: "
    f"{100 * correct / total:.2f}%"
)


print(
    f"Analyze questions: "
    f"{len(analyze_df)}"
)


print(
    f"Analyze Medium → Hard errors: "
    f"{len(medium_to_hard)}"
)


print(
    f"Analyze Hard → Medium errors: "
    f"{len(hard_to_medium)}"
)


print(
    f"Analyze Medium → Easy errors: "
    f"{len(medium_to_easy)}"
)


print()
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)