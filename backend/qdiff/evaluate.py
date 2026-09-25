import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from model import QDiffModel
from labels import (
    BLOOM_LABELS,
    DIFFICULTY_LABELS,
    BLOOM_TO_ID,
    DIFFICULTY_TO_ID
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TEST_FILE = "../qdiff_dataset/test.xlsx"
MODEL_PATH = "../qdiff_model/qdiff_model.pt"

BATCH_SIZE = 8
MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

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
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bloom_label": bloom_label,
            "difficulty_label": difficulty_label
        }


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_excel(TEST_FILE)

print("Test samples:", len(test_df))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "../qdiff_model"
)


# ============================================================
# DATA LOADER
# ============================================================

test_dataset = TestDataset(
    test_df,
    tokenizer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD MODEL
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


# ============================================================
# PREDICTIONS
# ============================================================

all_bloom_predictions = []
all_bloom_labels = []

all_difficulty_predictions = []
all_difficulty_labels = []


with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        bloom_logits, difficulty_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        bloom_predictions = torch.argmax(
            bloom_logits,
            dim=1
        ).cpu().numpy()

        difficulty_predictions = torch.argmax(
            difficulty_logits,
            dim=1
        ).cpu().numpy()

        all_bloom_predictions.extend(
            bloom_predictions
        )

        all_bloom_labels.extend(
            batch["bloom_label"].numpy()
        )

        all_difficulty_predictions.extend(
            difficulty_predictions
        )

        all_difficulty_labels.extend(
            batch["difficulty_label"].numpy()
        )


# ============================================================
# BLOOM METRICS
# ============================================================

bloom_accuracy = accuracy_score(
    all_bloom_labels,
    all_bloom_predictions
)

bloom_precision, bloom_recall, bloom_f1, _ = (
    precision_recall_fscore_support(
        all_bloom_labels,
        all_bloom_predictions,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# DIFFICULTY METRICS
# ============================================================

difficulty_accuracy = accuracy_score(
    all_difficulty_labels,
    all_difficulty_predictions
)

difficulty_precision, difficulty_recall, difficulty_f1, _ = (
    precision_recall_fscore_support(
        all_difficulty_labels,
        all_difficulty_predictions,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# DISPLAY BLOOM RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BLOOM'S TAXONOMY RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {bloom_accuracy:.4f}"
)

print(
    f"Precision : {bloom_precision:.4f}"
)

print(
    f"Recall    : {bloom_recall:.4f}"
)

print(
    f"F1-score  : {bloom_f1:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        all_bloom_labels,
        all_bloom_predictions,
        labels=list(range(len(BLOOM_LABELS))),
        target_names=BLOOM_LABELS,
        zero_division=0
    )
)


# ============================================================
# BLOOM CONFUSION MATRIX
# ============================================================

print("\nBloom Confusion Matrix:")

print(
    confusion_matrix(
        all_bloom_labels,
        all_bloom_predictions
    )
)


# ============================================================
# DISPLAY DIFFICULTY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DIFFICULTY RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {difficulty_accuracy:.4f}"
)

print(
    f"Precision : {difficulty_precision:.4f}"
)

print(
    f"Recall    : {difficulty_recall:.4f}"
)

print(
    f"F1-score  : {difficulty_f1:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        all_difficulty_labels,
        all_difficulty_predictions,
        labels=list(range(len(DIFFICULTY_LABELS))),
        target_names=DIFFICULTY_LABELS,
        zero_division=0
    )
)


# ============================================================
# DIFFICULTY CONFUSION MATRIX
# ============================================================

print("\nDifficulty Confusion Matrix:")

print(
    confusion_matrix(
        all_difficulty_labels,
        all_difficulty_predictions
    )
)


print("\nEvaluation completed successfully!")